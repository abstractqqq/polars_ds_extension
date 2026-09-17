//! Finite-window exponentially weighted least squares.
//!
//! For rho = 2^(-1 / half_life), maintain A = sum(w xx') and b = sum(w xy):
//! A[t] = rho A[t-1] + x[t]x[t]' - rho^W x[t-W]x[t-W]', and likewise for b.
//! Invalid rows contribute zero but retain their original position. See
//! maths/rolling_ewls.md for the derivation, rank policy and rebuilding costs.

use faer::{linalg::solvers::Solve, Mat, MatRef, Side};
use faer_traits::{math_utils::from_f64, RealField};
use num::Float;
use std::ops::Range;

struct CrossProducts<T: RealField + Float> {
    gram: Mat<T>,
    rhs: Vec<T>,
}

impl<T: RealField + Float> CrossProducts<T> {
    fn new(p: usize) -> Self {
        Self {
            gram: Mat::zeros(p, p),
            rhs: vec![T::zero(); p],
        }
    }

    fn scale(&mut self, factor: T) -> bool {
        if factor == T::zero() {
            // Fully decayed state contributes nothing, even after overflow.
            self.clear();
            return false;
        }
        let mut nonfinite = false;
        for i in 0..self.rhs.len() {
            self.rhs[i] *= factor;
            nonfinite |= !self.rhs[i].is_finite();
            for j in 0..=i {
                let value = *self.gram.get(i, j) * factor;
                nonfinite |= !value.is_finite();
                *self.gram.get_mut(i, j) = value;
                *self.gram.get_mut(j, i) = value;
            }
        }
        nonfinite
    }

    fn clear(&mut self) {
        self.rhs.fill(T::zero());
        for i in 0..self.rhs.len() {
            for j in 0..self.rhs.len() {
                *self.gram.get_mut(i, j) = T::zero();
            }
        }
    }

    fn add(&mut self, x: MatRef<T>, y: MatRef<T>, row: usize, weight: T) {
        if weight == T::zero() {
            return;
        }
        let target = *y.get(row, 0);
        for i in 0..self.rhs.len() {
            let xi = *x.get(row, i);
            self.rhs[i] += (weight * xi) * target;
            for j in 0..=i {
                let xj = *x.get(row, j);
                let value = *self.gram.get(i, j) + (weight * xi) * xj;
                *self.gram.get_mut(i, j) = value;
                *self.gram.get_mut(j, i) = value;
            }
        }
    }

    fn rebuild(
        &mut self,
        x: MatRef<T>,
        y: MatRef<T>,
        valid: &[bool],
        rows: Range<usize>,
        anchor: usize,
        half_life: f64,
    ) {
        self.clear();
        for (row, &is_valid) in valid.iter().enumerate().take(rows.end).skip(rows.start) {
            if is_valid {
                let weight = from_f64((-((anchor - row) as f64) / half_life).exp2());
                self.add(x, y, row, weight);
            }
        }
    }

    // A large outgoing observation can erase meaningful digits in a downdate.
    // Leave headroom for the solve and near-zero coefficients. The epsilon-based
    // threshold rebuilds earlier in f32, which loses more digits in subtraction.
    fn remove(&mut self, x: MatRef<T>, y: MatRef<T>, row: usize, weight: T) -> bool {
        if weight == T::zero() {
            return false;
        }
        let target = *y.get(row, 0);
        let cancellation_tol = T::epsilon().cbrt().max(from_f64(1e-4));
        let mut cancellation = false;
        for i in 0..self.rhs.len() {
            let xi = *x.get(row, i);
            let diagonal = *self.gram.get(i, i);
            let rhs = self.rhs[i];
            let new_diagonal = diagonal - (weight * xi) * xi;
            let new_rhs = rhs - (weight * xi) * target;
            cancellation |= !new_diagonal.is_finite()
                || !new_rhs.is_finite()
                || (diagonal != T::zero()
                    && new_diagonal.abs() < cancellation_tol * diagonal.abs())
                || (rhs != T::zero() && new_rhs.abs() < cancellation_tol * rhs.abs());
        }
        self.add(x, y, row, -weight);
        cancellation
    }

    fn solve(&self, rank_tol: T) -> Option<Mat<T>> {
        let p = self.rhs.len();
        let mut scales = Vec::with_capacity(p);
        for i in 0..p {
            let diagonal = *self.gram.get(i, i);
            if !diagonal.is_finite() || diagonal <= T::zero() || !self.rhs[i].is_finite() {
                return None;
            }
            scales.push(diagonal.sqrt());
        }
        // Diagonal scaling keeps the rank test invariant to column units. Use
        // successive divisions to avoid overflowing/underflowing scale[i]*scale[j].
        let normalized = Mat::from_fn(p, p, |i, j| *self.gram.get(i, j) / scales[i] / scales[j]);
        if !normalized.is_all_finite() {
            return None;
        }
        let factor = normalized.llt(Side::Lower).ok()?;
        // Same relative-determinant convention as faer_solve_lr_gated, evaluated
        // in log space and reusing the solve's single factorization.
        let log_det = factor
            .L()
            .diagonal()
            .column_vector()
            .iter()
            .fold(T::zero(), |acc, d| acc + (T::one() + T::one()) * d.ln());
        if !log_det.is_finite() || log_det <= rank_tol.ln() {
            return None;
        }
        let rhs = Mat::from_fn(p, 1, |i, _| self.rhs[i] / scales[i]);
        let mut coefficients = factor.solve(rhs);
        for (i, scale) in scales.into_iter().enumerate() {
            *coefficients.get_mut(i, 0) /= scale;
        }
        coefficients.is_all_finite().then_some(coefficients)
    }
}

/// Return one optional coefficient vector per input row, including warm-up nulls.
/// The plugin validates dimensions and parameters and supplies a joint X/y mask.
/// Accumulation and solving use the input precision, as in the other LR kernels.
pub fn faer_rolling_ewls<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    valid: &[bool],
    window: usize,
    min_rows: usize,
    half_life: f64,
    rank_tol: T,
) -> Vec<Option<Mat<T>>> {
    let n = x.nrows();
    let mut output = Vec::with_capacity(n);
    output.resize_with(n.min(window - 1), || None);
    if n < window {
        return output;
    }

    let decay: T = from_f64((-1.0 / half_life).exp2());
    // Compute expiry directly from the half-life, avoiding accumulated error
    // from raising a rounded decay factor to a large power.
    let expiry: T = from_f64((-(window as f64) / half_life).exp2());
    let mut products = CrossProducts::new(x.ncols());
    let mut count = valid[..window].iter().filter(|&&v| v).count();
    // Divide all weights by the newest valid row's weight. A common scale
    // cancels from unregularized least squares, so missing rows do not decay
    // the state into subnormal values. Original indices still control expiry
    // and the decay across a gap when the next valid row arrives.
    let mut anchor = valid[..window].iter().rposition(|&v| v).unwrap_or(0);
    let mut previous_fit = false;
    for t in window - 1..n {
        let previous_anchor = anchor;
        if valid[t] {
            anchor = t;
        }
        let mut rebuilt = (t + 1) % window == 0;
        if t >= window {
            count -= usize::from(valid[t - window]);
            count += usize::from(valid[t]);
            if !rebuilt {
                if anchor != previous_anchor {
                    let factor = if anchor - previous_anchor == 1 {
                        decay
                    } else {
                        from_f64((-((anchor - previous_anchor) as f64) / half_life).exp2())
                    };
                    rebuilt = products.scale(factor);
                }
                if valid[t - window] {
                    let weight = if anchor == t {
                        expiry
                    } else {
                        from_f64((-((anchor - (t - window)) as f64) / half_life).exp2())
                    };
                    rebuilt |= products.remove(x, y, t - window, weight);
                }
                if valid[t] {
                    products.add(x, y, t, T::one());
                }
            }
        }
        if count == 0 {
            products.clear();
        } else if rebuilt {
            // One O(W p^2) rebuild per W rows gives O(p^2) amortized work.
            products.rebuild(x, y, valid, t + 1 - window..t + 1, anchor, half_life);
        }
        if count < min_rows.max(x.ncols()) {
            output.push(None);
            previous_fit = false;
            continue;
        }
        let mut fit = products.solve(rank_tol);
        if fit.is_none() && previous_fit && !rebuilt {
            // Check a transition to degeneracy once against fresh statistics.
            // Persistently singular windows do not trigger an O(W) rescan per row.
            products.rebuild(x, y, valid, t + 1 - window..t + 1, anchor, half_life);
            fit = products.solve(rank_tol);
        }
        previous_fit = fit.is_some();
        output.push(fit);
    }
    output
}
