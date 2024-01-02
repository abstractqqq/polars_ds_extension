/// Checks whether or not the string is a stopword
/// Currently only supports English
use super::consts::EN_STOPWORDS;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Boolean)]
fn pl_is_stopword(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: BooleanChunked = ca.apply_nonnull_values_generic(DataType::Boolean, |s| {
        EN_STOPWORDS.binary_search(&s).is_ok()
    });
    Ok(out.into_series())
}
