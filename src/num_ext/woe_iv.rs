use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn woe_output(_: &[Field]) -> PolarsResult<Field> {
    let value = Field::new("value", DataType::String);
    let woe: Field = Field::new("woe", DataType::Float64);
    let v: Vec<Field> = vec![value, woe];
    Ok(Field::new("woe_output", DataType::Struct(v)))
}

fn iv_output(_: &[Field]) -> PolarsResult<Field> {
    let value = Field::new("value", DataType::String);
    let iv: Field = Field::new("iv", DataType::Float64);
    let v: Vec<Field> = vec![value, iv];
    Ok(Field::new("iv_output", DataType::Struct(v)))
}

/// Get a lazyframe needed to compute Weight Of Evidence.
/// Inputs[0] by default is the discrete bins / categories (cast to String at Python side)
/// Inputs[1] by default is the target (0s and 1s)
/// Nulls will be droppped
fn get_woe_frame(discrete_col: &Series, target: &Series) -> PolarsResult<LazyFrame> {
    let df = df!(
        "value" => discrete_col,
        "target" => target,
    )?;
    // Here we are adding 1 to make sure the event/non-event (goods/bads) are nonzero,
    // so that the computation will not yield inf as output.
    let out = df
        .lazy()
        .drop_nulls(None)
        .group_by([col("value")])
        .agg([len().alias("cnt"), col("target").sum().alias("goods")])
        .select([
            col("value"),
            ((col("goods") + lit(1)).cast(DataType::Float64)
                / (col("goods").sum() + lit(2)).cast(DataType::Float64))
            .alias("good_pct"),
            ((col("cnt") - col("goods") + lit(1)).cast(DataType::Float64)
                / (col("cnt").sum() - col("goods").sum() + lit(2)).cast(DataType::Float64))
            .alias("bad_pct"),
        ])
        .with_column(
            (col("good_pct") / col("bad_pct"))
                .log(std::f64::consts::E)
                .alias("woe"),
        );
    Ok(out)
}

/// WOE for each bin/category
#[polars_expr(output_type_func=woe_output)]
fn pl_woe_discrete(inputs: &[Series]) -> PolarsResult<Series> {
    let df = get_woe_frame(&inputs[0], &inputs[1])?
        .select([col("value"), col("woe")])
        .collect()?;

    Ok(df.into_struct("woe_output").into_series())
}

/// Information Value for each bin/category
/// The information value for this column/feature will be the sum.
#[polars_expr(output_type_func=iv_output)]
fn pl_iv(inputs: &[Series]) -> PolarsResult<Series> {
    let df = get_woe_frame(&inputs[0], &inputs[1])?
        .select([
            col("value"),
            ((col("good_pct") - col("bad_pct")) * col("woe")).alias("iv"),
        ])
        .collect()?;

    Ok(df.into_struct("iv_output").into_series())
}
