/// Powi using fast exponentiation.
/// Unfortunately, the pl.col("a").num_ext.powi(pl.col("b")) version may not
/// be faster, likely due to lack of SIMD (my hunch). However, something like
/// pl.col("a").num_ext.powi(16) is significantly faster than Polars's default.
use num::traits::Inv;
use polars::prelude::*;
use polars_core::prelude::arity::binary_elementwise_values;
use pyo3_polars::derive::polars_expr;

fn fast_exp_single(s: Series, n: i32) -> Series {
    if n == 0 {
        let ss = s.f64().unwrap();
        let out: Float64Chunked = ss.apply_values(|x| {
            if (x == 0.) | x.is_infinite() | x.is_nan() {
                f64::NAN
            } else {
                1.0
            }
        });
        return out.into_series();
    } else if n < 0 {
        return fast_exp_single(1.div(&s), -n);
    }

    let mut ss = s.clone();
    let mut m = n;
    let mut y = Series::from_vec("", vec![1_f64; s.len()]);
    while m > 0 {
        if m % 2 == 1 {
            y = &y * &ss;
        }
        ss = &ss * &ss;
        m >>= 1;
    }
    y
}

#[inline]
fn _fast_exp_pairwise(x: f64, n: u32) -> f64 {
    let mut m = n;
    let mut x = x;
    let mut y: f64 = 1.0;
    while m > 0 {
        if m % 2 == 1 {
            y *= x;
        }
        x *= x;
        m >>= 1;
    }
    y
}

#[inline] // Too many ifs and short circuits.
fn fast_exp_pairwise(x: f64, n: i32) -> f64 {
    if n == 0 {
        if x == 0. {
            // 0^0 is NaN
            return f64::NAN;
        } else {
            return 1.;
        }
    } else if n < 0 {
        return _fast_exp_pairwise(x.inv(), n.unsigned_abs());
    }
    _fast_exp_pairwise(x, n as u32)
}

#[polars_expr(output_type=Float64)]
fn pl_fast_exp(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs[0].clone();
    if !s.dtype().is_numeric() {
        return Err(PolarsError::ComputeError(
            "Input column type must be numeric.".into(),
        ));
    }

    let t = s.cast(&DataType::Float64)?;
    let exp = inputs[1].i32()?;

    if exp.len() == 1 {
        let n = exp.get(0).unwrap();
        Ok(fast_exp_single(t, n))
    } else if s.len() == exp.len() {
        let ca = t.f64()?;
        let out: Float64Chunked = binary_elementwise_values(ca, exp, fast_exp_pairwise);
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
