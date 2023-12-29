use inflections;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Utf8)]
fn pl_to_camel(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked =
        ca.apply_nonnull_values_generic(DataType::Utf8, inflections::case::to_camel_case);
    Ok(out.into_series())
}

#[polars_expr(output_type=Utf8)]
fn pl_to_snake(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked =
        ca.apply_nonnull_values_generic(DataType::Utf8, inflections::case::to_snake_case);
    Ok(out.into_series())
}

#[polars_expr(output_type=Utf8)]
fn pl_to_pascal(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked =
        ca.apply_nonnull_values_generic(DataType::Utf8, inflections::case::to_pascal_case);
    Ok(out.into_series())
}

#[polars_expr(output_type=Utf8)]
fn pl_to_constant(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked =
        ca.apply_nonnull_values_generic(DataType::Utf8, inflections::case::to_constant_case);
    Ok(out.into_series())
}
