use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float64)]
fn pl_conditional_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].name();
    let y = inputs[1].name();
    let out_name = format!("H({x}|{y})");
    let out_name = out_name.as_str();

    let df = DataFrame::new(inputs.to_vec())?;
    let mut out = df
        .lazy()
        .group_by([col(x), col(y)])
        .agg([count()])
        .with_columns([
            (col("count").sum().cast(DataType::Float64).over([col(y)])
                / col("count").sum().cast(DataType::Float64))
            .alias("p(y)"),
            (col("count").cast(DataType::Float64) / col("count").sum().cast(DataType::Float64))
                .alias("p(x,y)"),
        ])
        .select([(lit(-1.0_f64)
            * ((col("p(x,y)") / col("p(y)"))
                .log(std::f64::consts::E)
                .dot(col("p(x,y)"))))
        .alias(out_name)])
        .collect()?;

    out.drop_in_place(out_name)
}
