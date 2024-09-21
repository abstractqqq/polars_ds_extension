/// GCD and LCM for integers in dataframe.
use polars::prelude::*;
use polars_core::prelude::arity::binary_elementwise_values;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Int32)]
fn pl_gcd(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i32()?;
    let ca2 = inputs[1].i32()?;
    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out: Int32Chunked = ca1.apply_values(|a| num::integer::gcd(a, b));
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Int32Chunked = binary_elementwise_values(ca1, ca2, num::integer::gcd);
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=Int32)]
fn pl_lcm(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i32()?;
    let ca2 = inputs[1].i32()?;
    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out: Int32Chunked = ca1.apply_values(|a| num::integer::lcm(a, b));
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Int32Chunked = binary_elementwise_values(ca1, ca2, num::integer::lcm);
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
