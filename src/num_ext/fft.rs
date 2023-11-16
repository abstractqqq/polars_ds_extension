use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rustfft::FftPlanner;

fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    let real = Field::new("re", DataType::Float64);
    let complex = Field::new("im", DataType::Float64);
    let v: Vec<Field> = vec![real, complex];
    Ok(Field::new("complex", DataType::Struct(v)))
}

#[polars_expr(output_type_func=complex_output)]
fn pl_fft(inputs: &[Series]) -> PolarsResult<Series> {
    // Take a step argument

    let s = inputs[0].clone();
    if s.null_count() > 0 {
        return Err(PolarsError::ComputeError(
            "FFT input cannot have null values.".into(),
        ));
    }

    let s = s.cast(&DataType::Float64)?;
    let s = s.f64()?;
    let mut buf: Vec<num::complex::Complex64> = s.into_no_null_iter().map(|x| x.into()).collect();

    let name = s.name();
    let forward = inputs[1].bool()?;
    let forward = forward.get(0).unwrap();

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = if forward {
        planner.plan_fft_forward(buf.len())
    } else {
        planner.plan_fft_inverse(buf.len())
    };
    fft.process(&mut buf);

    let mut re_builder = PrimitiveChunkedBuilder::<Float64Type>::new("re", buf.len());
    let mut im_builder = PrimitiveChunkedBuilder::<Float64Type>::new("im", buf.len());
    for c in buf {
        re_builder.append_value(c.re);
        im_builder.append_value(c.im);
    }

    let re: Series = re_builder.finish().into();
    let im: Series = im_builder.finish().into();

    let fft_struct = StructChunked::new(name, &[re, im])?;

    Ok(fft_struct.into_series())
}
