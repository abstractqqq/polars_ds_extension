use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

fn target_encode_output(_: &[Field]) -> PolarsResult<Field> {
    let values = Field::new("values", DataType::String);
    let value = Field::new("value", DataType::Float64);
    let v: Vec<Field> = vec![values, value];
    Ok(Field::new("target_encoded", DataType::Struct(v)))
}

#[derive(Deserialize, Debug)]
pub(crate) struct TargetEncodeKwargs {
    pub(crate) min_samples_leaf: f64,
    pub(crate) smoothing: f64,
}

#[inline(always)]
fn get_target_encode_frame(
    discrete_col: &Series,
    target: &Series,
    target_mean: f64,
    min_samples_leaf: f64,
    smoothing: f64,
) -> PolarsResult<LazyFrame> {
    let df = df!(
        "values" => discrete_col.cast(&DataType::String)?,
        "target" => target
    )?;

    Ok(df
        .lazy()
        .group_by([col("values")])
        .agg([len().alias("cnt"), col("target").mean().alias("cond_p")])
        .with_column(
            (lit(1f64)
                / (lit(1f64)
                    + ((-(col("cnt").cast(DataType::Float64) - lit(min_samples_leaf))
                        / lit(smoothing))
                    .exp())))
            .alias("alpha"),
        )
        .select([
            col("values"),
            (col("alpha") * col("cond_p") + (lit(1f64) - col("alpha")) * lit(target_mean))
                .alias("to"),
        ]))
}

#[polars_expr(output_type_func=target_encode_output)]
fn pl_target_encode(inputs: &[Series], kwargs: TargetEncodeKwargs) -> PolarsResult<Series> {
    // Inputs[0] and inputs[1] are the string column and the target respectively

    let target_mean = inputs[2].f64()?;
    let target_mean = target_mean.get(0).unwrap();

    let min_samples_leaf = kwargs.min_samples_leaf;
    let smoothing = kwargs.smoothing;

    let encoding_frame = get_target_encode_frame(
        &inputs[0],
        &inputs[1],
        target_mean,
        min_samples_leaf,
        smoothing,
    )?
    .collect()?;

    Ok(encoding_frame.into_struct("target_encoded").into_series())
}
