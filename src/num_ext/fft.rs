use super::complex::complex_output;
/// Performs forward FFT.
/// Since data in dataframe are always real numbers, only realfft
/// is implemented and inverse fft is not implemented and even if it
/// is eventually implemented, it would likely not be a dataframe
/// operation.
use itertools::Either;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use realfft::RealFftPlanner;

#[polars_expr(output_type_func=complex_output)]
fn pl_rfft(inputs: &[Series]) -> PolarsResult<Series> {
    // Take a step argument

    let s = inputs[0].f64()?;
    let length = inputs[1].u32()?;
    let length = length.get(0).unwrap_or(s.len() as u32) as usize;

    if length > s.len() {
        return Err(PolarsError::ComputeError(
            "FFT: Length should not be bigger than length of input.".into(),
        ));
    }

    let input_vec: Result<Vec<f64>, PolarsError> = match s.to_vec_null_aware() {
        Either::Left(v) => Ok(v),
        Either::Right(_) => Err(PolarsError::ComputeError(
            "FFT: Input should not contain nulls.".into(),
        )),
    };
    let mut input_vec = input_vec?;

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(length);

    let mut spectrum = r2c.make_output_vec();
    let _ = r2c.process(&mut input_vec, &mut spectrum);

    let mut builder =
        ListPrimitiveChunkedBuilder::<Float64Type>::new("complex", length, 2, DataType::Float64);
    for c in spectrum {
        builder.append_slice(&[c.re, c.im])
    }

    let out = builder.finish();
    Ok(out.into_series())
}
