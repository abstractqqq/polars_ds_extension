use crate::float_output;
/// The logit, expit and gamma function as defined in SciPy
use num::traits::Float;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[inline]
fn logit<T: Float>(x: T) -> T {
    if x.is_zero() {
        T::neg_infinity()
    } else if x.is_one() {
        T::infinity()
    } else if x < T::zero() || x > T::one() {
        T::nan()
    } else {
        (x / (T::one() - x)).ln()
    }
}

#[inline]
fn expit<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

#[polars_expr(output_type_func=float_output)]
fn pl_logit(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float64 => {
            // will only allocate a new Series when type != f64
            let ss = s.cast(&DataType::Float64)?;
            let ca = ss.f64().unwrap();
            let out = ca.apply_values(logit);
            Ok(out.into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            let out = ca.apply_values(logit);
            Ok(out.into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_expit(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float64 => {
            // will only allocate a new Series when type != f64
            let ss = s.cast(&DataType::Float64)?;
            let ca = ss.f64().unwrap();
            let out = ca.apply_values(expit);
            Ok(out.into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            let out = ca.apply_values(expit);
            Ok(out.into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type=Float64)]
fn pl_gamma(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float64 => {
            // will only allocate a new Series when type != f64
            let ss = s.cast(&DataType::Float64)?;
            let ca = ss.f64().unwrap();
            let out = ca.apply_values(f64::gamma);
            Ok(out.into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            let out = ca.apply_values(f32::gamma);
            Ok(out.into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}
