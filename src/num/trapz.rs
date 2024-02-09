/// Integration via Trapezoidal rule.
use ndarray::{s, ArrayView1};
use polars::{
    prelude::{PolarsError, PolarsResult},
    series::Series,
};
use pyo3_polars::derive::polars_expr;

#[inline(always)]
pub fn trapz(y: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let y_s = &y.slice(s![1..]) + &y.slice(s![..-1]);
    let x_d = &x.slice(s![1..]) - &x.slice(s![..-1]);

    0.5 * (x_d
        .into_iter()
        .zip(y_s.into_iter())
        .fold(0., |acc, (x, y)| acc + x * y))
}

pub fn trapz_dx(y: ArrayView1<f64>, dx: f64) -> f64 {
    let s = y.slice(s![1..-1]).sum();
    let ss = 0.5 * (y.get(0).unwrap_or(&0.) + y.last().unwrap_or(&0.));
    dx * (s + ss)
}

#[polars_expr(output_type=Float64)]
fn pl_trapz(inputs: &[Series]) -> PolarsResult<Series> {
    let y = inputs[0].f64()?.rechunk();
    let y = y.to_ndarray()?;
    let x = inputs[1].f64()?;
    if x.null_count() > 0 {
        return Err(PolarsError::ComputeError(
            "For trapezoidal integration to work, x axis must not contain nulls.".into(),
        ));
    }
    if x.len() == 1 {
        let dx = x.get(0).unwrap();
        Ok(Series::from_iter([trapz_dx(y, dx)]))
    } else if x.len() == y.len() {
        let x = x.rechunk();
        let x = x.to_ndarray()?;
        Ok(Series::from_iter([trapz(y, x)]))
    } else {
        Err(PolarsError::ComputeError(
            "Input must have the same length.".into(),
        ))
    }
}
