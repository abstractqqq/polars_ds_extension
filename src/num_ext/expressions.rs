use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use num;

#[polars_expr(output_type=Int64)]
fn pl_gcd(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i64()?;
    let ca2 = inputs[1].i64()?;

    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out:Int64Chunked = ca1.into_iter().map(|op_a| {
            if let Some(a) = op_a {
                Some(num::integer::gcd(a, b))
            } else {
                None 
            }
        }).collect();
        Ok(out.into_series())

    } else if ca1.len() == ca2.len() {
        let out: Int64Chunked = ca1.into_iter()
        .zip(ca2.into_iter())
        .map(|(op_a, op_b)| {
            if let (Some(a), Some(b)) = (op_a, op_b) {
                Some(num::integer::gcd(a,b))
            } else {
                None
            }
        }).collect();
        Ok(out.into_series())

    } else {
        Err(PolarsError::ComputeError("Inputs must have the same length.".into()))
    }
}

#[polars_expr(output_type=Int64)]
fn pl_lcm(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i64()?;
    let ca2 = inputs[1].i64()?;

    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out:Int64Chunked = ca1.into_iter().map(|op_a| {
            if let Some(a) = op_a {
                Some(num::integer::lcm(a, b))
            } else {
                None 
            }
        }).collect();
        Ok(out.into_series())

    } else if ca1.len() == ca2.len() {
        let out: Int64Chunked = ca1.into_iter()
        .zip(ca2.into_iter())
        .map(|(op_a, op_b)| {
            if let (Some(a), Some(b)) = (op_a, op_b) {
                Some(num::integer::lcm(a,b))
            } else {
                None
            }
        }).collect();
        Ok(out.into_series())

    } else {
        Err(PolarsError::ComputeError("Inputs must have the same length.".into()))
    }
}