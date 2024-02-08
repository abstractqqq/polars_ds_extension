use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn woe_output(_: &[Field]) -> PolarsResult<Field> {
    let values = Field::new("values", DataType::String);
    let woe: Field = Field::new("woe", DataType::Float64);
    let v: Vec<Field> = vec![values, woe];
    Ok(Field::new("woe_output", DataType::Struct(v)))
}

fn iv_output(_: &[Field]) -> PolarsResult<Field> {
    let values = Field::new("values", DataType::String);
    let woe: Field = Field::new("iv", DataType::Float64);
    let v: Vec<Field> = vec![values, woe];
    Ok(Field::new("iv_output", DataType::Struct(v)))
}

/// Get a lazyframe needed to compute WOE.
/// Inputs[0] by default is the target (0s and 1s)
/// Inputs[1] by default is the discrete bins / categories
fn get_woe_frame(inputs: &[Series]) -> PolarsResult<LazyFrame> {
    let categories = &inputs[1].cast(&DataType::String)?;
    let df = df!(
        "target" => inputs[0].clone(),
        "values" => categories
    )?;
    // Here we are adding 1 to make sure the event/non-event (goods/bads) are nonzero,
    // so that the computation will not yield inf as output.
    let out = df
        .lazy()
        .group_by([col("values")])
        .agg([len().alias("cnt"), col("target").sum().alias("goods")])
        .select([
            col("values"),
            ((col("goods") + lit(1)).cast(DataType::Float64)
                / (col("goods").sum() + lit(2)).cast(DataType::Float64))
            .alias("good_pct"),
            ((col("cnt") - col("goods") + lit(1)).cast(DataType::Float64)
                / (col("cnt").sum() - col("goods").sum() + lit(2)).cast(DataType::Float64))
            .alias("bad_pct"),
        ])
        .with_column(
            (col("bad_pct") / col("good_pct"))
                .log(std::f64::consts::E)
                .alias("woe"),
        );
    Ok(out)
}

/// WOE for each bin/category
#[polars_expr(output_type_func=woe_output)]
fn pl_woe_discrete(inputs: &[Series]) -> PolarsResult<Series> {
    let df = get_woe_frame(inputs)?
        .select([col("values"), col("woe")])
        .collect()?;

    let out = df.into_struct("woe_output");
    Ok(out.into_series())
}

/// Information Value for each bin/category
/// The information value for this column/feature will be the sum.
#[polars_expr(output_type_func=iv_output)]
fn pl_iv(inputs: &[Series]) -> PolarsResult<Series> {
    let df = get_woe_frame(inputs)?
        .select([
            col("values"),
            ((col("bad_pct") - col("good_pct")) * col("woe")).alias("iv"),
        ])
        .collect()?;

    let out = df.into_struct("iv_output");
    Ok(out.into_series())
}
