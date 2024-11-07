/// Integration via Trapezoidal rule.
use cfavml;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[inline(always)]
pub fn trapz(y: &[f64], x: &[f64]) -> f64 {
    let mut y_s = vec![0.; y.len() - 1];
    cfavml::add_vector(&y[1..], &y[..y.len() - 1], &mut y_s);
    let mut x_d = vec![0.; y.len() - 1];
    cfavml::sub_vector(&x[1..], &x[..x.len() - 1], &mut x_d);
    0.5 * cfavml::dot(&y_s, &x_d)
}

pub fn trapz_dx(y: &[f64], dx: f64) -> f64 {
    let s = y[1..y.len() - 1].iter().sum::<f64>();
    let ss = 0.5 * (y.get(0).unwrap_or(&0.) + y.last().unwrap_or(&0.));
    dx * (s + ss)
}

#[polars_expr(output_type=Float64)]
fn pl_trapz(inputs: &[Series]) -> PolarsResult<Series> {
    let y = inputs[0].f64()?;
    let x = inputs[1].f64()?;
    if y.len() < 2 {
        let ca = Float64Chunked::from_slice("".into(), &[f64::NAN]);
        return Ok(ca.into_series());
    }
    if x.has_nulls() || y.has_nulls() {
        return Err(PolarsError::ComputeError(
            "For trapezoidal integration to work, x and y must not contain nulls.".into(),
        ));
    }
    let y = y.cont_slice()?;
    if x.len() == 1 {
        let dx = x.get(0).unwrap();
        let ca = Float64Chunked::from_slice("".into(), &[trapz_dx(y, dx)]);
        Ok(ca.into_series())
    } else if x.len() == y.len() {
        let x = x.cont_slice()?;
        let ca = Float64Chunked::from_slice("".into(), &[trapz(y, x)]);
        Ok(ca.into_series())
    } else {
        Err(PolarsError::ComputeError(
            "Input must have the same length or x must be a scalar.".into(),
        ))
    }
}
