use crate::utils::list_u32_output;
use num::Integer;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn first_digit<T: Integer + Copy>(u: T) -> T {
    let ten = (0..10).fold(T::zero(), |acc: T, _| acc + T::one());
    let mut v = u;
    let mut d = T::zero();
    while v != T::zero() {
        d = v % ten;
        v = v / ten;
    }
    d
}

#[polars_expr(output_type_func=list_u32_output)]
fn pl_benford_law(inputs: &[Series]) -> PolarsResult<Series> {
    let mut out = [0; 10];
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => {
            let ss = s.u8().unwrap();
            for x in ss.into_no_null_iter() {
                let d = first_digit(x);
                out[d as usize] += 1;
            }
        }
        DataType::UInt16 => {
            let ss = s.u16().unwrap();
            for x in ss.into_no_null_iter() {
                let d = first_digit(x);
                out[d as usize] += 1;
            }
        }
        DataType::UInt32 => {
            let ss = s.u32().unwrap();
            for x in ss.into_no_null_iter() {
                let d = first_digit(x);
                out[d as usize] += 1;
            }
        }
        DataType::UInt64 => {
            let ss = s.u64().unwrap();
            for x in ss.into_no_null_iter() {
                let d = first_digit(x);
                out[d as usize] += 1;
            }
        }
        DataType::Int8 => {
            let ss = s.i8().unwrap();
            for x in ss.into_no_null_iter() {
                let d = first_digit(x); // could be negative
                out[d.abs() as usize] += 1;
            }
        }
        DataType::Int16 => {
            let ss = s.i16().unwrap();
            for x in ss.into_no_null_iter() {
                let d = first_digit(x); // could be negative
                out[d.abs() as usize] += 1;
            }
        }
        DataType::Int32 => {
            let ss = s.i32().unwrap();
            for x in ss.into_no_null_iter() {
                let d = first_digit(x); // could be negative
                out[d.abs() as usize] += 1;
            }
        }
        DataType::Int64 => {
            let ss = s.i64().unwrap();
            for x in ss.into_no_null_iter() {
                let d = first_digit(x); // could be negative
                out[d.abs() as usize] += 1;
            }
        }
        DataType::Float32 => {
            let ss = s.f32().unwrap();
            for x in ss.into_no_null_iter() {
                if x.is_finite() {
                    let x_char = x
                        .abs()
                        .to_string()
                        .chars()
                        .find(|c| *c != '0' && *c != '.')
                        .unwrap_or('0');
                    let idx = x_char.to_digit(10).unwrap() as usize;
                    out[idx] += 1;
                }
            }
        }
        DataType::Float64 => {
            let ss = s.f64().unwrap();
            for x in ss.into_no_null_iter() {
                if x.is_finite() {
                    let x_char = x
                        .abs()
                        .to_string()
                        .chars()
                        .find(|c| *c != '0' && *c != '.')
                        .unwrap_or('0');
                    let idx = x_char.to_digit(10).unwrap() as usize;
                    out[idx] += 1;
                }
            }
        }
        _ => return Err(PolarsError::ComputeError("Invalid incoming type.".into())),
    }

    let mut list_builder: ListPrimitiveChunkedBuilder<UInt32Type> =
        ListPrimitiveChunkedBuilder::new("first_digit_count", 1, 9, DataType::UInt32);

    list_builder.append_slice(&out[1..]);
    let out = list_builder.finish();
    Ok(out.into_series())
}
