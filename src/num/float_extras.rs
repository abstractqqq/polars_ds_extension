/// Extra functions for floating points
/// trunc, fract, exp2, logit, expit, gamma,
/// The logit, expit and gamma functions are as defined in SciPy
use crate::utils::{first_field_output, float_output};
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

fn cast_and_apply_logit<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(logit::<f64>)
}

fn cast_and_apply_expit<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(expit::<f64>)
}

fn cast_and_apply_gamma<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(f64::gamma)
}

fn cast_and_apply_exp2<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(f64::exp2)
}

fn cast_and_apply_sign<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(f64::signum)
}

#[polars_expr(output_type_func=float_output)]
fn pl_logit(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => Ok(cast_and_apply_logit(s.u8().unwrap()).into_series()),
        DataType::UInt16 => Ok(cast_and_apply_logit(s.u16().unwrap()).into_series()),
        DataType::UInt32 => Ok(cast_and_apply_logit(s.u32().unwrap()).into_series()),
        DataType::UInt64 => Ok(cast_and_apply_logit(s.u64().unwrap()).into_series()),
        DataType::Int8 => Ok(cast_and_apply_logit(s.i8().unwrap()).into_series()),
        DataType::Int16 => Ok(cast_and_apply_logit(s.i16().unwrap()).into_series()),
        DataType::Int32 => Ok(cast_and_apply_logit(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(cast_and_apply_logit(s.i64().unwrap()).into_series()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(logit).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(logit).into_series())
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
        DataType::UInt8 => Ok(cast_and_apply_expit(s.u8().unwrap()).into_series()),
        DataType::UInt16 => Ok(cast_and_apply_expit(s.u16().unwrap()).into_series()),
        DataType::UInt32 => Ok(cast_and_apply_expit(s.u32().unwrap()).into_series()),
        DataType::UInt64 => Ok(cast_and_apply_expit(s.u64().unwrap()).into_series()),
        DataType::Int8 => Ok(cast_and_apply_expit(s.i8().unwrap()).into_series()),
        DataType::Int16 => Ok(cast_and_apply_expit(s.i16().unwrap()).into_series()),
        DataType::Int32 => Ok(cast_and_apply_expit(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(cast_and_apply_expit(s.i64().unwrap()).into_series()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(expit).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(expit).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_gamma(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => Ok(cast_and_apply_gamma(s.u8().unwrap()).into_series()),
        DataType::UInt16 => Ok(cast_and_apply_gamma(s.u16().unwrap()).into_series()),
        DataType::UInt32 => Ok(cast_and_apply_gamma(s.u32().unwrap()).into_series()),
        DataType::UInt64 => Ok(cast_and_apply_gamma(s.u64().unwrap()).into_series()),
        DataType::Int8 => Ok(cast_and_apply_gamma(s.i8().unwrap()).into_series()),
        DataType::Int16 => Ok(cast_and_apply_gamma(s.i16().unwrap()).into_series()),
        DataType::Int32 => Ok(cast_and_apply_gamma(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(cast_and_apply_gamma(s.i64().unwrap()).into_series()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(f64::gamma).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(f32::gamma).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_exp2(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => Ok(cast_and_apply_exp2(s.u8().unwrap()).into_series()),
        DataType::UInt16 => Ok(cast_and_apply_exp2(s.u16().unwrap()).into_series()),
        DataType::UInt32 => Ok(cast_and_apply_exp2(s.u32().unwrap()).into_series()),
        DataType::UInt64 => Ok(cast_and_apply_exp2(s.u64().unwrap()).into_series()),
        DataType::Int8 => Ok(cast_and_apply_exp2(s.i8().unwrap()).into_series()),
        DataType::Int16 => Ok(cast_and_apply_exp2(s.i16().unwrap()).into_series()),
        DataType::Int32 => Ok(cast_and_apply_exp2(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(cast_and_apply_exp2(s.i64().unwrap()).into_series()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(f64::exp2).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(f32::exp2).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type_func=first_field_output)]
fn pl_trunc(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => Ok(s.clone()),
        DataType::UInt16 => Ok(s.clone()),
        DataType::UInt32 => Ok(s.clone()),
        DataType::UInt64 => Ok(s.clone()),
        DataType::Int8 => Ok(s.clone()),
        DataType::Int16 => Ok(s.clone()),
        DataType::Int32 => Ok(s.clone()),
        DataType::Int64 => Ok(s.clone()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(f64::trunc).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(f32::trunc).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type_func=first_field_output)]
fn pl_fract(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => Ok(Series::from_vec(s.name(), vec![0_u8; s.len()])),
        DataType::UInt16 => Ok(Series::from_vec(s.name(), vec![0_u16; s.len()])),
        DataType::UInt32 => Ok(Series::from_vec(s.name(), vec![0_u32; s.len()])),
        DataType::UInt64 => Ok(Series::from_vec(s.name(), vec![0_u64; s.len()])),
        DataType::Int8 => Ok(Series::from_vec(s.name(), vec![0_i8; s.len()])),
        DataType::Int16 => Ok(Series::from_vec(s.name(), vec![0_i16; s.len()])),
        DataType::Int32 => Ok(Series::from_vec(s.name(), vec![0_i32; s.len()])),
        DataType::Int64 => Ok(Series::from_vec(s.name(), vec![0_i64; s.len()])),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(f64::fract).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(f32::fract).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_signum(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => Ok(Series::from_vec(s.name(), vec![1_f64; s.len()])),
        DataType::UInt16 => Ok(Series::from_vec(s.name(), vec![1_f64; s.len()])),
        DataType::UInt32 => Ok(Series::from_vec(s.name(), vec![1_f64; s.len()])),
        DataType::UInt64 => Ok(Series::from_vec(s.name(), vec![1_f64; s.len()])),
        DataType::Int8 => Ok(cast_and_apply_sign(s.i8().unwrap()).into_series()),
        DataType::Int16 => Ok(cast_and_apply_sign(s.i16().unwrap()).into_series()),
        DataType::Int32 => Ok(cast_and_apply_sign(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(cast_and_apply_sign(s.i64().unwrap()).into_series()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(f64::signum).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(f32::signum).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}
