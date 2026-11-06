//! Bounded QR reduction followed by truncated-SVD least squares.

use faer::{Mat, MatRef, Par};
use faer_traits::RealField;
use num::Float;

/// Truncated-SVD least squares with bounded Householder QR reduction.
/// Ridge uses [X; sqrt(lambda) D], where D excludes the trailing intercept.
pub fn faer_solve_lr_rcond_stable<T: RealField + Float>(
    x: MatRef<'_, T>,
    y: MatRef<T>,
    lambda: T,
    add_bias: bool,
    rcond: T,
) -> Result<(Mat<T>, Vec<T>), String> {
    solve_with_options(x, y, lambda, add_bias, rcond, RcondOptions::DEFAULT)
}

// Internal tuning controls, also exercised by the native validation harness.
// They are deliberately absent from the Python API.
#[derive(Clone, Copy)]
struct RcondOptions {
    block_rows: usize,
    qr_target_exponent: Option<i32>,
    rescale_each_block: bool,
}

impl RcondOptions {
    const DEFAULT: Self = Self {
        block_rows: 512,
        // Workaround for subnormal intermediates in faer 8377404e78
        // (linalg/reductions/norm_l2.rs); revalidate when updating faer.
        qr_target_exponent: Some(256),
        rescale_each_block: false,
    };
}

fn solve_with_options<T: RealField + Float>(
    x: MatRef<'_, T>,
    y: MatRef<'_, T>,
    lambda: T,
    add_bias: bool,
    rcond: T,
    options: RcondOptions,
) -> Result<(Mat<T>, Vec<T>), String> {
    use faer::linalg::householder;

    let (nrows, ncols) = x.shape();
    let targets = y.ncols();
    if ncols == 0 {
        return Ok((Mat::zeros(0, targets), Vec::new()));
    }
    // Promote only the bounded workspaces. In particular, Float32 keeps its
    // public cutoff floor, supplied by the caller, without a full Float64 copy.
    let x_max = rcond_max_abs(x)?;
    let y_max = rcond_max_abs(y)?;
    let conversion_error = || "unsupported regression precision".to_owned();
    let penalty = if lambda > T::zero() {
        lambda.to_f64().ok_or_else(conversion_error)?.sqrt()
    } else {
        0.0
    };
    if !penalty.is_finite() {
        return Err("positive Ridge penalty must be finite".into());
    }
    let x_max = x_max.max(penalty);
    if x_max == 0.0 {
        return Ok((Mat::zeros(ncols, targets), vec![T::zero(); ncols]));
    }
    let x_exponent = rcond_scale_exponent(x_max);
    let y_exponent = rcond_scale_exponent(y_max);
    // Keep carried R in the same units as the input blocks. The global shift
    // avoids a scan and two extra scaling passes per block.
    let design_exponent = x_exponent
        - if options.rescale_each_block {
            0
        } else {
            options.qr_target_exponent.unwrap_or(0)
        };
    let y_scale = rcond_power_of_two(y_exponent);
    let block_rows = options.block_rows.max(ncols).max(1);
    let workspace_rows = ncols + block_rows.min(nrows);
    let mut reduced = Mat::<f64>::zeros(ncols, ncols);
    let mut reduced_y = Mat::<f64>::zeros(ncols, targets);
    if penalty > 0.0 {
        for i in 0..ncols.saturating_sub(usize::from(add_bias)) {
            reduced[(i, i)] = rcond_rescale(penalty, -design_exponent);
        }
    }
    let mut block = Mat::<f64>::zeros(workspace_rows, ncols + targets);
    let x_multiplier = rcond_rescale(1.0, -design_exponent);
    let y_multiplier = 1.0 / y_scale;
    let mut scratch = faer::dyn_stack::MemBuffer::new(
        householder::apply_block_householder_on_the_left_in_place_scratch::<f64>(
            workspace_rows,
            1,
            ncols + targets,
        ),
    );
    for start in (0..nrows).step_by(block_rows) {
        let rows = block_rows.min(nrows - start);
        let used_rows = ncols + rows;
        block
            .as_mut()
            .submatrix_mut(0, 0, ncols, ncols)
            .copy_from(&reduced);
        block
            .as_mut()
            .submatrix_mut(0, ncols, ncols, targets)
            .copy_from(&reduced_y);
        // Merge the scale scan into the input copy; carried state is small.
        let mut maximum_bits = 0u64;
        if options.rescale_each_block {
            for column in reduced.col_iter() {
                for &value in column.iter() {
                    maximum_bits = maximum_bits.max(value.to_bits() & 0x7fff_ffff_ffff_ffff);
                }
            }
        }
        for j in 0..ncols {
            for i in 0..rows {
                let value = x[(start + i, j)].to_f64().ok_or_else(conversion_error)?;
                let value = if x_multiplier.is_finite() && x_multiplier != 0.0 {
                    value * x_multiplier
                } else {
                    rcond_rescale(value, -design_exponent)
                };
                block[(ncols + i, j)] = value;
                if options.rescale_each_block {
                    maximum_bits = maximum_bits.max(value.to_bits() & 0x7fff_ffff_ffff_ffff);
                }
            }
        }
        for j in 0..targets {
            for i in 0..rows {
                block[(ncols + i, ncols + j)] =
                    y[(start + i, j)].to_f64().ok_or_else(conversion_error)? * y_multiplier;
            }
        }
        let maximum = f64::from_bits(maximum_bits);
        if options.rescale_each_block && maximum == 0.0 {
            continue;
        }
        // Move the block's maximum near 2^256. The pinned faer's robust
        // norms accumulate three scales, whose small-scale intermediate values
        // would otherwise be subnormal for ordinary unit-scale inputs.
        // Restore the exact power-of-two scale before carrying R forward.
        let qr_exponent = if options.rescale_each_block {
            options
                .qr_target_exponent
                .map_or(0, |target| target - rcond_scale_exponent(maximum))
        } else {
            0
        };
        if qr_exponent == 0 {
            // Global scaling was already fused into the input copy.
        } else if qr_exponent <= 1023 {
            let scale = rcond_power_of_two(qr_exponent);
            for column in block
                .as_mut()
                .submatrix_mut(0, 0, used_rows, ncols)
                .col_iter_mut()
            {
                for value in column.iter_mut() {
                    *value *= scale;
                }
            }
        } else {
            for column in block
                .as_mut()
                .submatrix_mut(0, 0, used_rows, ncols)
                .col_iter_mut()
            {
                for value in column.iter_mut() {
                    *value = rcond_rescale(*value, qr_exponent);
                }
            }
        }
        rcond_qr_reduce(
            block.as_mut().subrows_mut(0, used_rows),
            ncols,
            faer::dyn_stack::MemStack::new(&mut scratch),
        );
        for j in 0..ncols {
            for i in 0..ncols {
                reduced[(i, j)] = if i <= j {
                    rcond_rescale(block[(i, j)], -qr_exponent)
                } else {
                    0.0
                };
            }
        }
        reduced_y.copy_from(block.as_ref().submatrix(0, ncols, ncols, targets));
    }
    if !options.rescale_each_block {
        // Restore unit-scale R once before SVD. Leaving the extra QR shift in
        // S would underflow tiny target projections during division, even when
        // the final coefficient is representable after undoing X/y scaling.
        let shift = options.qr_target_exponent.unwrap_or(0);
        for column in reduced.col_iter_mut() {
            for value in column.iter_mut() {
                *value = rcond_rescale(*value, -shift);
            }
        }
    }
    let svd = reduced
        .thin_svd()
        .map_err(|_| "svd algorithm did not converge".to_owned())?;
    let spectrum = svd.S().column_vector();
    let mut singular_values = Vec::with_capacity(ncols);
    for &sigma in spectrum.iter() {
        let value = T::from(rcond_rescale(sigma, x_exponent)).ok_or_else(conversion_error)?;
        if !Float::is_finite(value) {
            return Err("regression singular values exceed configured precision".into());
        }
        singular_values.push(value);
    }
    // Use the returned precision for the cutoff too. Otherwise Float32 output
    // can round a retained direction onto the boundary and imply a different rank.
    let threshold = rcond * singular_values[0];
    let mut projected = svd.U().transpose() * reduced_y;
    for column in projected.col_iter_mut() {
        for (i, value) in column.iter_mut().enumerate() {
            *value = if singular_values[i] > threshold {
                *value / spectrum[i]
            } else {
                0.0
            };
        }
    }
    let solution = svd.V() * projected;
    let mut coefficients = Mat::<T>::zeros(ncols, targets);
    for j in 0..targets {
        for i in 0..ncols {
            coefficients[(i, j)] =
                T::from(rcond_rescale(solution[(i, j)], y_exponent - x_exponent))
                    .ok_or_else(conversion_error)?;
        }
    }
    Ok((coefficients, singular_values))
}

// GEQR2-style Householder steps, using faer primitives. Every feature gets
// its own reflector position, even for an exactly zero column; numerical rank
// is determined only by the final SVD. The trailing target columns receive the
// same transformations, without materializing Q or searching for column pivots.
fn rcond_qr_reduce(
    mut block: faer::MatMut<'_, f64>,
    ncols: usize,
    stack: &mut faer::dyn_stack::MemStack,
) {
    use faer::linalg::householder;
    let used_rows = block.nrows();
    for j in 0..ncols {
        let (mut leading, mut trailing) = block.as_mut().split_at_col_mut(j + 1);
        let info = {
            let mut column = leading.as_mut().col_mut(j);
            let (mut head, tail) = column
                .as_mut()
                .subrows_mut(j, used_rows - j)
                .split_at_row_mut(1);
            householder::make_householder_in_place(&mut head[0], tail)
        };
        if info.tau.is_infinite() {
            continue;
        }
        let tau = [info.tau];
        householder::apply_block_householder_on_the_left_in_place_with_conj(
            leading.as_ref().submatrix(j, j, used_rows - j, 1),
            MatRef::from_column_major_slice(&tau, 1, 1),
            faer::Conj::No,
            trailing.as_mut().subrows_mut(j, used_rows - j),
            Par::Seq,
            stack,
        );
    }
}

// Powers of two preserve relative singular values and avoid squaring a scale.
// Subnormal maxima use MIN_POSITIVE: dividing by it still gives safe QR norms.
fn rcond_scale_exponent(maximum: f64) -> i32 {
    if maximum == 0.0 {
        0
    } else {
        (((maximum.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-1022)
    }
}

fn rcond_power_of_two(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

fn rcond_rescale(mut value: f64, mut exponent: i32) -> f64 {
    // Staging avoids overflow/underflow in the scale itself when X and y have
    // very different magnitudes, even if the final coefficient is representable.
    while exponent > 1023 {
        value *= rcond_power_of_two(1023);
        exponent -= 1023;
    }
    while exponent < -1022 {
        value *= rcond_power_of_two(-1022);
        exponent += 1022;
    }
    value * rcond_power_of_two(exponent)
}

// For nonnegative IEEE-754 values, integer bit order is magnitude order.
// Infinity and NaNs sort above every finite value, allowing a single final
// validation instead of floating-point max/finite branches for each element.
fn rcond_max_abs<T: RealField + Float>(matrix: MatRef<'_, T>) -> Result<f64, String> {
    let mut maximum_bits = 0u64;
    for column in matrix.col_iter() {
        for &value in column.iter() {
            let bits = value
                .to_f64()
                .ok_or_else(|| "unsupported regression precision".to_owned())?
                .to_bits()
                & 0x7fff_ffff_ffff_ffff;
            maximum_bits = maximum_bits.max(bits);
        }
    }
    if maximum_bits >= f64::INFINITY.to_bits() {
        return Err("regression inputs must be finite after null handling".into());
    }
    Ok(f64::from_bits(maximum_bits))
}

#[cfg(test)]
#[path = "rcond_tests.rs"]
mod tests;
