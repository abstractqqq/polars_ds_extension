use itertools::Either;
// use num::{complex::Complex64, Zero};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
// use rustfft::FftPlanner;
use realfft::RealFftPlanner;

fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "complex",
        DataType::List(Box::new(DataType::Float64)),
    ))
}

// #[polars_expr(output_type_func=complex_output)]
// fn pl_fft(inputs: &[Series]) -> PolarsResult<Series> {
//     // Take a step argument

//     let s = inputs[0].clone();
//     if s.null_count() > 0 {
//         return Err(PolarsError::ComputeError(
//             "FFT input cannot have null values.".into(),
//         ));
//     }

//     let s = s.cast(&DataType::Float64)?;
//     let s = s.f64()?;
//     let mut buf: Vec<num::complex::Complex64> = s.into_no_null_iter().map(|x| x.into()).collect();

//     let name = s.name();
//     let forward = inputs[1].bool()?;
//     let forward = forward.get(0).unwrap();

//     let mut planner: FftPlanner<f64> = FftPlanner::new();
//     let fft = if forward {
//         planner.plan_fft_forward(buf.len())
//     } else {
//         planner.plan_fft_inverse(buf.len())
//     };
//     fft.process(&mut buf);

//     let mut re_builder = PrimitiveChunkedBuilder::<Float64Type>::new("re", buf.len());
//     let mut im_builder = PrimitiveChunkedBuilder::<Float64Type>::new("im", buf.len());
//     for c in buf {
//         re_builder.append_value(c.re);
//         im_builder.append_value(c.im);
//     }

//     let re: Series = re_builder.finish().into();
//     let im: Series = im_builder.finish().into();

//     let fft_struct = StructChunked::new(name, &[re, im])?;

//     Ok(fft_struct.into_series())
// }

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
