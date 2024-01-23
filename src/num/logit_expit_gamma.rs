/// The logit, expit and gamma function as defined in SciPy
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use num::traits::Float;

pub fn float_output(fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(fields).map_to_float_dtype()
}

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
        | DataType::Int64 => {
            let ss = s.cast(&DataType::Float32)?;
            let ca = ss.f32()?;
            let out = ca.apply_values(logit);
            Ok(out.into_series())
        },
        DataType::Float32 => {
            let ca = s.f32()?;
            let out = ca.apply_values(logit);
            Ok(out.into_series())
        },
        DataType::Float64 => {
            let ca = s.f64()?;
            let out = ca.apply_values(logit);
            Ok(out.into_series())
        },
        _ => {
            Err(PolarsError::ShapeMismatch(
                "Input column must be numerical.".into(),
            ))
        }
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
        | DataType::Int64 => {
            let ss = s.cast(&DataType::Float32)?;
            let ca = ss.f32()?;
            let out = ca.apply_values(expit);
            Ok(out.into_series())
        },
        DataType::Float32 => {
            let ca = s.f32()?;
            let out = ca.apply_values(expit);
            Ok(out.into_series())
        },
        DataType::Float64 => {
            let ca = s.f64()?;
            let out = ca.apply_values(expit);
            Ok(out.into_series())
        },
        _ => {
            Err(PolarsError::ShapeMismatch(
                "Input column must be numerical.".into(),
            ))
        }
    }
}

#[polars_expr(output_type=Float64)]
fn pl_gamma(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    // Gamma is not a trait. So always cast.
    let ss = s.cast(&DataType::Float64)?;
    let ca = ss.f64()?;
    let out = ca.apply_values(|x| x.gamma());
    Ok(out.into_series())
}
