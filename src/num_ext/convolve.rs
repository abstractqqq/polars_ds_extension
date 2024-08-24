use cfavml;
use ndarray::Array1;
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

fn valid_fft_convolve(input: &[f64], filter: &[f64]) -> PolarsResult<Vec<f64>> {
    let in_shape = input.len();

    // Prepare
    let mut output_vec = vec![0.; in_shape];
    output_vec[..in_shape].copy_from_slice(input);

    let mut oth = vec![0.; in_shape];
    oth[..filter.len()].copy_from_slice(filter);

    let mut planner: RealFftPlanner<f64> = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(in_shape);
    let c2r = planner.plan_fft_inverse(in_shape);
    // let mut spec_p = r2c.make_output_vec();
    let mut spec_p = Array1::from_vec(r2c.make_output_vec());
    // let mut spec_q = r2c.make_output_vec();
    let mut spec_q = Array1::from_vec(r2c.make_output_vec());
    // Forward FFT on the inputs
    let _ = r2c.process(&mut output_vec, spec_p.as_slice_mut().unwrap());
    // .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let _ = r2c.process(&mut oth, spec_q.as_slice_mut().unwrap());
    // .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    // After forward FFT, multiply elementwise
    spec_p = spec_p * spec_q;
    // Inverse FFT
    let _ = c2r.process(spec_p.as_slice_mut().unwrap(), &mut output_vec);
    // .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    Ok(output_vec)
}

fn fft_convolve(input: &[f64], filter: &[f64], mode: ConvMode) -> PolarsResult<Vec<f64>> {
    match mode {
        ConvMode::FULL => {
            let t = filter.len() - 1;
            let mut padded_input = vec![0.; input.len() + 2 * t];
            let from_to = t..(t + input.len());
            padded_input[from_to].copy_from_slice(input);
            fft_convolve(&padded_input, filter, ConvMode::VALID)
        }
        ConvMode::SAME => {
            let skip = (filter.len() - 1) / 2;
            let out = fft_convolve(input, filter, ConvMode::FULL)?;
            Ok(out.into_iter().skip(skip).take(input.len()).collect())
        }
        ConvMode::LEFT => {
            let n = input.len();
            let mut out = fft_convolve(input, filter, ConvMode::FULL)?;
            out.truncate(n);
            Ok(out)
        }
        ConvMode::RIGHT => {
            let out = fft_convolve(input, filter, ConvMode::FULL)?;
            Ok(out.into_iter().skip(filter.len() - 1).collect())
        }
        ConvMode::VALID => {
            let out = valid_fft_convolve(input, filter)?;
            let n = out.len() as f64;
            Ok(out
                .into_iter()
                .skip(filter.len() - 1)
                .map(|x| x / n)
                .collect())
        }
    }
}

fn convolve(
    input: &[f64],
    filter: &[f64],
    mode: ConvMode,
    parallel: bool,
) -> PolarsResult<Vec<f64>> {
    match mode {
        ConvMode::FULL => {
            let t = filter.len() - 1;
            let mut padded_input = vec![0.; input.len() + 2 * t];
            let from_to = t..(t + input.len());
            padded_input[from_to].copy_from_slice(input);
            convolve(&padded_input, filter, ConvMode::VALID, parallel)
        }
        ConvMode::SAME => {
            let skip = (filter.len() - 1) / 2;
            let out = convolve(input, filter, ConvMode::FULL, parallel)?;
            Ok(out.into_iter().skip(skip).take(input.len()).collect())
        }
        ConvMode::LEFT => {
            let n = input.len();
            let mut out = convolve(input, filter, ConvMode::FULL, parallel)?;
            out.truncate(n);
            Ok(out)
        }
        ConvMode::RIGHT => {
            let out = convolve(input, filter, ConvMode::FULL, parallel)?;
            Ok(out.into_iter().skip(filter.len() - 1).collect())
        }
        ConvMode::VALID => {
            if parallel {
                let mut out = vec![0f64; input.len() - filter.len() + 1];
                input
                    .par_windows(filter.len())
                    .map(|sl| cfavml::dot(filter, sl))
                    .collect_into_vec(&mut out);
                Ok(out)
            } else {
                Ok(input
                    .windows(filter.len())
                    .map(|sl| cfavml::dot(filter, sl))
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
            "Convolution: The filter should have smaller length than the input column, and filter should have length >= 2.".into(),
        ));
    }

    let input = s1.cont_slice().unwrap();
    let filter = s2.cont_slice().unwrap();

    let out = match method {
        ConvMethod::FFT => fft_convolve(input, filter, mode),
        ConvMethod::DIRECT => convolve(input, filter, mode, par),
    }?;
    let ca = Float64Chunked::from_vec(s1.name(), out);
    Ok(ca.into_series())
}
