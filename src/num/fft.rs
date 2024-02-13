/// Performs forward FFT.
/// Since data in dataframe are always real numbers, only realfft
/// is implemented and. 5-10x slower than NumPy for small data (~ a few thousands rows)
/// but is slighly faster once data gets bigger.
use crate::utils::complex_output;
use itertools::Either;
use num::complex::Complex64;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use realfft::RealFftPlanner;

#[polars_expr(output_type_func=complex_output)]
fn pl_rfft(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs[0].f64()?;
    let n = inputs[1].u32()?;
    let mut n = n.get(0).unwrap_or(s.len() as u32) as usize;
    let return_full = inputs[2].bool()?;
    let return_full = return_full.get(0).unwrap_or(false);

    let mut input_vec = match s.to_vec_null_aware() {
        Either::Left(v) => Ok(v),
        Either::Right(_) => Err(PolarsError::ComputeError(
            "FFT: Input should not contain nulls.".into(),
        )),
    }?;

    if n > input_vec.len() {
        input_vec.extend(vec![0.; n.abs_diff(input_vec.len())]);
    } else if n < input_vec.len() {
        input_vec.truncate(n);
    }
    let input_len = input_vec.len();

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(input_len);

    let mut spectrum: Vec<Complex64> = r2c.make_output_vec();
    let _ = r2c.process(&mut input_vec, &mut spectrum);

    n = if return_full {
        input_vec.len() // full length
    } else {
        spectrum.len() // simplified output of rfft
    };

    let mut builder =
        ListPrimitiveChunkedBuilder::<Float64Type>::new("complex", n, 2, DataType::Float64);

    if return_full {
        for c in spectrum.iter() {
            builder.append_slice(&[c.re, c.im])
        }
        if input_len % 2 == 0 {
            let take_n = (input_len >> 1).abs_diff(1);
            for c in spectrum.into_iter().rev().skip(1).take(take_n) {
                builder.append_slice(&[c.re, -c.im]);
            }
        } else {
            let take_n = input_len >> 1;
            for c in spectrum.into_iter().rev().take(take_n) {
                builder.append_slice(&[c.re, -c.im]);
            }
        }
    } else {
        for c in spectrum {
            builder.append_slice(&[c.re, c.im])
        }
    }

    let out = builder.finish();
    out.cast(&DataType::Array(Box::new(DataType::Float64), 2))
    // Ok(out.into_series())
}
