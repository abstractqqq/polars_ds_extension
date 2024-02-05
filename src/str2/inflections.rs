use inflections;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=String)]
fn pl_to_camel(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_values(|s| inflections::case::to_camel_case(s).into());
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn pl_to_snake(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_values(|s| inflections::case::to_snake_case(s).into());
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn pl_to_pascal(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_values(|s| inflections::case::to_pascal_case(s).into());
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn pl_to_constant(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_values(|s| inflections::case::to_constant_case(s).into());
    Ok(out.into_series())
}
