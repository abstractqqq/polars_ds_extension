use cfavml;
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::utils::rayon::{
        iter::{IndexedParallelIterator, ParallelIterator},
        slice::ParallelSlice,
    },
};

use realfft::RealFftPlanner;
use serde::Deserialize;

// Small vec optimizations? Fixed sized allocation for <= 4096?

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

// fn next_pow_2(n:usize) -> usize {
//     let mut m:usize = 2;
//     while m < n {
//         m <<= 1;
//     }
//     m
// }

#[inline]
fn valid_fft_convolve(input: &[f64], kernel: &[f64]) -> PolarsResult<Vec<f64>> {
    let in_shape = input.len();

    // Prepare
    let mut output_vec = vec![0.; in_shape];
    output_vec[..in_shape].copy_from_slice(input);

    let mut planner: RealFftPlanner<f64> = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(in_shape);
    let c2r = planner.plan_fft_inverse(in_shape);

    let mut spec_p = r2c.make_output_vec();
    let mut spec_q = r2c.make_output_vec();

    // Forward FFT on the inputs
    let _ = r2c.process(&mut output_vec, &mut spec_p);

    // Write kernel to output_vec, then 0 fill the rest
    output_vec[..kernel.len()].copy_from_slice(kernel);
    for i in kernel.len()..output_vec.len() {
        output_vec[i] = 0.;
    }
    // Now output_vec is the kernel
    let _ = r2c.process(&mut output_vec, &mut spec_q);

    // After forward FFT, multiply elementwise
    for i in 0..spec_p.len() {
        spec_p[i] *= spec_q[i];
    }
    // Inverse FFT
    let _ = c2r.process(&mut spec_p, &mut output_vec);

    Ok(output_vec)
}

fn fft_convolve(input: &[f64], kernel: &[f64], mode: ConvMode) -> PolarsResult<Vec<f64>> {
    match mode {
        ConvMode::FULL => {
            let t = kernel.len() - 1;
            let mut padded_input = vec![0.; input.len() + 2 * t];
            let from_to = t..(t + input.len());
            padded_input[from_to].copy_from_slice(input);
            fft_convolve(&padded_input, kernel, ConvMode::VALID)
        }
        ConvMode::SAME => {
            let skip = (kernel.len() - 1) / 2;
            let out = fft_convolve(input, kernel, ConvMode::FULL)?;
            Ok(out.into_iter().skip(skip).take(input.len()).collect())
        }
        ConvMode::LEFT => {
            let n = input.len();
            let mut out = fft_convolve(input, kernel, ConvMode::FULL)?;
            out.truncate(n);
            Ok(out)
        }
        ConvMode::RIGHT => {
            let out = fft_convolve(input, kernel, ConvMode::FULL)?;
            Ok(out.into_iter().skip(kernel.len() - 1).collect())
        }
        ConvMode::VALID => {
            let out = valid_fft_convolve(input, kernel)?;
            let n = out.len() as f64;
            Ok(out
                .into_iter()
                .skip(kernel.len() - 1)
                .map(|x| x / n)
                .collect())
        }
    }
}

fn convolve(
    input: &[f64],
    kernel: &[f64],
    mode: ConvMode,
    parallel: bool,
) -> PolarsResult<Vec<f64>> {
    match mode {
        ConvMode::FULL => {
            let t = kernel.len() - 1;
            let mut padded_input = vec![0.; input.len() + 2 * t];
            let from_to = t..(t + input.len());
            padded_input[from_to].copy_from_slice(input);
            convolve(&padded_input, kernel, ConvMode::VALID, parallel)
        }
        ConvMode::SAME => {
            let skip = (kernel.len() - 1) / 2;
            let out = convolve(input, kernel, ConvMode::FULL, parallel)?;
            Ok(out.into_iter().skip(skip).take(input.len()).collect())
        }
        ConvMode::LEFT => {
            let n = input.len();
            let mut out = convolve(input, kernel, ConvMode::FULL, parallel)?;
            out.truncate(n);
            Ok(out)
        }
        ConvMode::RIGHT => {
            let out = convolve(input, kernel, ConvMode::FULL, parallel)?;
            Ok(out.into_iter().skip(kernel.len() - 1).collect())
        }
        ConvMode::VALID => {
            if parallel {
                let mut out = vec![0f64; input.len() - kernel.len() + 1];
                input
                    .par_windows(kernel.len())
                    .map(|sl| cfavml::dot(kernel, sl))
                    .collect_into_vec(&mut out);
                Ok(out)
            } else {
                Ok(input
                    .windows(kernel.len())
                    .map(|sl| cfavml::dot(kernel, sl))
                    .collect())
            }
        }
    }
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
