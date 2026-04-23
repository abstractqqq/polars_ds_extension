use crate::utils::list_u32_output;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// Optimized first digit for integers using only division.
/// For primitives, using a literal 10 allows the compiler to optimize division
/// into multiplication by a magic constant.
macro_rules! impl_benford_int {
    ($ss:expr, $out:expr) => {
        for mut v in $ss.into_no_null_iter() {
            if v == 0 { continue; }
            while v >= 10 {
                v /= 10;
            }
            $out[v as usize] += 1;
        }
    };
    ($ss:expr, $out:expr, signed) => {
        for x in $ss.into_no_null_iter() {
            let mut v = x.unsigned_abs();
            if v == 0 { continue; }
            while v >= 10 {
                v /= 10;
            }
            $out[v as usize] += 1;
        }
    };
}

#[polars_expr(output_type_func=list_u32_output)]
fn pl_benford_law(inputs: &[Series]) -> PolarsResult<Series> {
    let mut out = [0u32; 10];
    let s = &inputs[0];
    match s.dtype() {
        DataType::UInt8 => {
            let ss = s.u8().unwrap();
            impl_benford_int!(ss, out);
        }
        DataType::UInt16 => {
            let ss = s.u16().unwrap();
            impl_benford_int!(ss, out);
        }
        DataType::UInt32 => {
            let ss = s.u32().unwrap();
            impl_benford_int!(ss, out);
        }
        DataType::UInt64 => {
            let ss = s.u64().unwrap();
            impl_benford_int!(ss, out);
        }
        DataType::Int8 => {
            let ss = s.i8().unwrap();
            impl_benford_int!(ss, out, signed);
        }
        DataType::Int16 => {
            let ss = s.i16().unwrap();
            impl_benford_int!(ss, out, signed);
        }
        DataType::Int32 => {
            let ss = s.i32().unwrap();
            impl_benford_int!(ss, out, signed);
        }
        DataType::Int64 => {
            let ss = s.i64().unwrap();
            impl_benford_int!(ss, out, signed);
        }
        DataType::Float32 => {
            let ss = s.f32().unwrap();
            for x in ss.into_no_null_iter() {
                if x.is_finite() && x != 0.0 {
                    let abs_x = x.abs() as f64;
                    let log10 = abs_x.log10().floor();
                    let d = (abs_x / 10.0_f64.powf(log10)).floor() as usize;
                    out[d.min(9)] += 1;
                }
            }
        }
        DataType::Float64 => {
            let ss = s.f64().unwrap();
            for x in ss.into_no_null_iter() {
                if x.is_finite() && x != 0.0 {
                    let abs_x = x.abs();
                    let log10 = abs_x.log10().floor();
                    let d = (abs_x / 10.0_f64.powf(log10)).floor() as usize;
                    out[d.min(9)] += 1;
                }
            }
        }
        _ => return Err(PolarsError::ComputeError("Invalid incoming type.".into())),
    }

    let mut list_builder: ListPrimitiveChunkedBuilder<UInt32Type> =
        ListPrimitiveChunkedBuilder::new("first_digit_count".into(), 1, 9, DataType::UInt32);

    list_builder.append_slice(&out[1..]);
    let out = list_builder.finish();
    Ok(out.into_series())
}
