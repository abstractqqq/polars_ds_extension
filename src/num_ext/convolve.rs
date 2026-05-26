use crate::utils::dot_product;
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::utils::rayon,
    export::polars_core::utils::rayon::{
        iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
        slice::{ParallelSlice, ParallelSliceMut},
    },
};
use realfft::RealFftPlanner;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct ConvolveKwargs {
    pub(crate) mode: String,
    pub(crate) method: String,
    pub(crate) parallel: bool,
}

enum ConvMode {
    FULL,
    SAME,
    LEFT,
    RIGHT,
    VALID,
}

impl TryFrom<String> for ConvMode {
    type Error = PolarsError;
    fn try_from(value: String) -> PolarsResult<Self> {
        match value.to_lowercase().as_ref() {
            "full" => Ok(Self::FULL),
            "same" => Ok(Self::SAME),
            "left" => Ok(Self::LEFT),
            "right" => Ok(Self::RIGHT),
            "valid" => Ok(Self::VALID),
            _ => Err(PolarsError::ComputeError(
                "Unknown convolution mode.".into(),
            )),
        }
    }
}

enum ConvMethod {
    FFT,
    DIRECT,
}

impl TryFrom<String> for ConvMethod {
    type Error = PolarsError;
    fn try_from(value: String) -> PolarsResult<Self> {
        match value.to_lowercase().as_ref() {
            "fft" => Ok(Self::FFT),
            "direct" => Ok(Self::DIRECT),
            _ => Err(PolarsError::ComputeError(
                "Unknown convolution method.".into(),
            )),
        }
    }
}

#[inline]
fn fft_convolve(input: &[f64], kernel: &[f64], mode: ConvMode) -> PolarsResult<Vec<f64>> {
    let n = input.len();
    let m = kernel.len();
    let l = n + m - 1;
    let p = l.next_power_of_two();

    let mut planner: RealFftPlanner<f64> = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(p);
    let c2r = planner.plan_fft_inverse(p);

    let mut spec_p = r2c.make_output_vec();
    let mut spec_q = r2c.make_output_vec();

    // FFT for input
    let mut padded_input = vec![0.0; p];
    padded_input[..n].copy_from_slice(input);
    let _ = r2c.process(&mut padded_input, &mut spec_p);

    // FFT for kernel
    let mut padded_kernel = vec![0.0; p];
    padded_kernel[..m].copy_from_slice(kernel);
    let _ = r2c.process(&mut padded_kernel, &mut spec_q);

    // Multiply spectrum
    spec_p.iter_mut().zip(spec_q.iter()).for_each(|(x, y)| {
        *x *= y;
    });

    // Inverse FFT
    let mut output_vec = vec![0.0; p];
    let _ = c2r.process(&mut spec_p, &mut output_vec);

    // Normalization and slicing
    let scale = 1.0 / p as f64;
    let (skip, take) = match mode {
        ConvMode::FULL => (0, l),
        ConvMode::SAME => ((m - 1) / 2, n),
        ConvMode::LEFT => (0, n),
        ConvMode::RIGHT => (m - 1, n),
        ConvMode::VALID => (m - 1, n - m + 1),
    };

    Ok(output_vec
        .into_iter()
        .skip(skip)
        .take(take)
        .map(|x| x * scale)
        .collect())
}

fn convolve(
    input: &[f64],
    kernel: &[f64],
    mode: ConvMode,
    parallel: bool,
) -> PolarsResult<Vec<f64>> {
    let n = input.len();
    let m = kernel.len();

    let (out_len, skip) = match mode {
        ConvMode::FULL => (n + m - 1, 0),
        ConvMode::SAME => (n, (m - 1) / 2),
        ConvMode::LEFT => (n, 0),
        ConvMode::RIGHT => (n, m - 1),
        ConvMode::VALID => (n - m + 1, m - 1),
    };

    let mut out = vec![0f64; out_len];

    // center region is where j + skip >= m - 1 AND j + skip <= n - 1
    // j >= m - 1 - skip  (saturating at 0)
    // j <= n - 1 - skip
    let left_end = (m - 1).saturating_sub(skip).min(out_len);
    let right_start = n.saturating_sub(skip).max(left_end).min(out_len);

    // Left edge
    for j in 0..left_end {
        let mut sum = 0.0;
        for k in 0..m {
            let p_idx = j + skip + k;
            if p_idx >= m - 1 && p_idx - (m - 1) < n {
                sum += kernel[k] * input[p_idx - (m - 1)];
            }
        }
        out[j] = sum;
    }

    // Center
    if right_start > left_end {
        let center_out = &mut out[left_end..right_start];
        let in_start = left_end + skip - (m - 1);
        let in_end = in_start + center_out.len() + m - 1;
        let center_in = &input[in_start..in_end];

        if parallel {
            center_out
                .par_iter_mut()
                .zip(center_in.par_windows(m))
                .for_each(|(o, w)| {
                    *o = dot_product(kernel, w);
                });
        } else {
            center_out
                .iter_mut()
                .zip(center_in.windows(m))
                .for_each(|(o, w)| {
                    *o = dot_product(kernel, w);
                });
        }
    }

    // Right edge
    for j in right_start..out_len {
        let mut sum = 0.0;
        for k in 0..m {
            let p_idx = j + skip + k;
            if p_idx >= m - 1 && p_idx - (m - 1) < n {
                sum += kernel[k] * input[p_idx - (m - 1)];
            }
        }
        out[j] = sum;
    }

    Ok(out)
}

#[polars_expr(output_type=Float64)]
fn pl_convolve(
    inputs: &[Series],
    context: CallerContext,
    kwargs: ConvolveKwargs,
) -> PolarsResult<Series> {
    let s1 = inputs[0].f64()?;
    let s2 = inputs[1].f64()?;

    let mode: ConvMode = kwargs.mode.try_into()?;
    let method: ConvMethod = kwargs.method.try_into()?;
    let par = kwargs.parallel && !context.parallel();

    if s1.len() < s2.len() || s2.len() < 2 {
        return Err(PolarsError::ComputeError(
            "Convolution: The kernel should have smaller length than the input column, and kernel should have length >= 2.".into(),
        ));
    }

    let input = s1.cont_slice().unwrap();
    let kernel = s2.cont_slice().unwrap();

    let out = match method {
        ConvMethod::FFT => fft_convolve(input, kernel, mode),
        ConvMethod::DIRECT => convolve(input, kernel, mode, par),
    }?;
    let ca = Float64Chunked::from_vec(s1.name().clone(), out);
    Ok(ca.into_series())
}
