/// The logit, expit and gamma function as defined in SciPy
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float64)]
fn pl_logit(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ss = s.cast(&DataType::Float64)?;
    let ca = ss.f64()?;
    let out = ca.apply_values(|x| {
        if x == 0. {
            f64::NEG_INFINITY
        } else if x == 1. {
            f64::INFINITY
        } else if x < 0. || x > 1. {
            f64::NAN
        } else {
            (x / (1. - x)).ln()
        }
    });
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_expit(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ss = s.cast(&DataType::Float64)?;
    let ca = ss.f64()?;
    let out = ca.apply_values(|x| 1.0 / ((-x).exp() + 1.0));
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_gamma(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let ss = s.cast(&DataType::Float64)?;
    let ca = ss.f64()?;
    let out = ca.apply_values(|x| x.gamma());
    Ok(out.into_series())
}
