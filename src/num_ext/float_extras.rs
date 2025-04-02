use crate::stats_utils::gamma::digamma;
/// Extra functions for floating points
/// trunc, fract, exp2, logit, expit, gamma,
/// The logit, expit and gamma functions are as defined in SciPy
use crate::utils::{first_field_output, float_output};
use arity::binary_elementwise_values;
use num::traits::Float;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

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

fn xlogy<T: Float>(x: T, y: T) -> T {
    if x == T::zero() && !y.is_nan() {
        T::zero()
    } else {
        x * y.ln()
    }
}

fn expit<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

fn cast_and_apply_logit<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float32Chunked {
    ca.cast_and_apply_in_place(logit::<f32>)
}

fn cast_and_apply_expit<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float32Chunked {
    ca.cast_and_apply_in_place(expit::<f32>)
}

fn cast_and_apply_gamma<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float32Chunked {
    ca.cast_and_apply_in_place(f32::gamma)
}

fn cast_and_apply_exp2<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float32Chunked {
    ca.cast_and_apply_in_place(f32::exp2)
}

fn cast_and_apply_next_up<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float32Chunked {
    ca.cast_and_apply_in_place(f32::next_up)
}

fn cast_and_apply_next_down<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float32Chunked {
    ca.cast_and_apply_in_place(f32::next_down)
}

fn cast_and_apply_diagamma<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
    ca.cast_and_apply_in_place(digamma)
}

// fn cast_and_apply_sign<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Float64Chunked {
//     ca.cast_and_apply_in_place(f64::signum)
// }

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

#[polars_expr(output_type_func=float_output)]
fn pl_fract(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];

    if s.dtype().is_integer() {
        Ok(Series::from_vec(s.name().clone(), vec![0f32; s.len()]))
    } else {
        if s.dtype() == &DataType::Float64 {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(f64::fract).into_series())
        } else if s.dtype() == &DataType::Float32 {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(f32::fract).into_series())
        } else {
            Err(PolarsError::ComputeError(
                "Input column must be numerical.".into(),
            ))
        }
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_next_up(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];

    match s.dtype() {
        DataType::UInt8 => Ok(cast_and_apply_next_up(s.u8().unwrap()).into_series()),
        DataType::UInt16 => Ok(cast_and_apply_next_up(s.u16().unwrap()).into_series()),
        DataType::UInt32 => Ok(cast_and_apply_next_up(s.u32().unwrap()).into_series()),
        DataType::UInt64 => Ok(cast_and_apply_next_up(s.u64().unwrap()).into_series()),
        DataType::Int8 => Ok(cast_and_apply_next_up(s.i8().unwrap()).into_series()),
        DataType::Int16 => Ok(cast_and_apply_next_up(s.i16().unwrap()).into_series()),
        DataType::Int32 => Ok(cast_and_apply_next_up(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(cast_and_apply_next_up(s.i64().unwrap()).into_series()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(f64::next_up).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(f32::next_up).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_next_down(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];

    match s.dtype() {
        DataType::UInt8 => Ok(cast_and_apply_next_down(s.u8().unwrap()).into_series()),
        DataType::UInt16 => Ok(cast_and_apply_next_down(s.u16().unwrap()).into_series()),
        DataType::UInt32 => Ok(cast_and_apply_next_down(s.u32().unwrap()).into_series()),
        DataType::UInt64 => Ok(cast_and_apply_next_down(s.u64().unwrap()).into_series()),
        DataType::Int8 => Ok(cast_and_apply_next_down(s.i8().unwrap()).into_series()),
        DataType::Int16 => Ok(cast_and_apply_next_down(s.i16().unwrap()).into_series()),
        DataType::Int32 => Ok(cast_and_apply_next_down(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(cast_and_apply_next_down(s.i64().unwrap()).into_series()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(f64::next_down).into_series())
        }
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            Ok(ca.apply_values(f32::next_down).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_xlogy(inputs: &[Series]) -> PolarsResult<Series> {
    let s1 = &inputs[0];
    let s2 = &inputs[1];
    let ca1 = s1.f64().unwrap();
    let ca2 = s2.f64().unwrap();
    let ca: Float64Chunked = binary_elementwise_values(ca1, ca2, xlogy::<f64>);
    Ok(ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_diagamma(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => Ok(cast_and_apply_diagamma(s.u8().unwrap()).into_series()),
        DataType::UInt16 => Ok(cast_and_apply_diagamma(s.u16().unwrap()).into_series()),
        DataType::UInt32 => Ok(cast_and_apply_diagamma(s.u32().unwrap()).into_series()),
        DataType::UInt64 => Ok(cast_and_apply_diagamma(s.u64().unwrap()).into_series()),
        DataType::Int8 => Ok(cast_and_apply_diagamma(s.i8().unwrap()).into_series()),
        DataType::Int16 => Ok(cast_and_apply_diagamma(s.i16().unwrap()).into_series()),
        DataType::Int32 => Ok(cast_and_apply_diagamma(s.i32().unwrap()).into_series()),
        DataType::Int64 => Ok(cast_and_apply_diagamma(s.i64().unwrap()).into_series()),
        DataType::Float32 => Ok(cast_and_apply_diagamma(s.f32().unwrap()).into_series()),
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            Ok(ca.apply_values(digamma).into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be numerical.".into(),
        )),
    }
}
