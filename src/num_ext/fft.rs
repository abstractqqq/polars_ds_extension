use rustfft::FftPlanner;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    let real = Field::new("re", DataType::Float64);
    let complex = Field::new("im", DataType::Float64);
    let v: Vec<Field> = vec![real, complex];
    Ok(Field::new("complex", DataType::Struct(v)))
}

#[polars_expr(output_type_func=complex_output)]
fn pl_fft(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs[0].clone();
    if s.null_count() > 0 {
        return Err(PolarsError::ComputeError(
            "FFT Input cannot have null values.".into(),
        ));
    }
    let mut buf: Vec<num::complex::Complex64> = match s.dtype() {
        DataType::Float64 => s.f64()?.into_no_null_iter().map(|x| x.into()).collect(),
        DataType::Float32 => {
            let temp = s.cast(&DataType::Float64)?;
            temp.f64()?.into_no_null_iter().map(|x| x.into()).collect()
        }
        _ => {
            return Err(PolarsError::ComputeError(
                "FFT Input must be floats.".into(),
            ))
        }
    };

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

    let mut re: Vec<f64> = Vec::with_capacity(buf.len());
    let mut im: Vec<f64> = Vec::with_capacity(buf.len());
    for c in buf {
        re.push(c.re);
        im.push(c.im);
    }

    let fft_struct = df!(
        "re" => re,
        "im" => im,
    )?
    .into_struct(name)
    .into_series();

    Ok(fft_struct)
}