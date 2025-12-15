use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn kaplan_meier_prob(_: &[Field]) -> PolarsResult<Field> {
    let t = Field::new("t".into(), DataType::Float64);
    let p = Field::new("prob".into(), DataType::Float64);
    let v: Vec<Field> = vec![t, p];
    Ok(Field::new("kaplan_meier".into(), DataType::Struct(v)))
}

#[polars_expr(output_type_func=kaplan_meier_prob)]
fn pl_kaplan_meier(inputs: &[Series]) -> PolarsResult<Series> {

    let n1 = inputs[0].len();
    let n2 = inputs[1].len();

    if n1 != n2 {
        return Err(PolarsError::ShapeMismatch("Length of status column is not the same as the length of survival time column.".into()));
    }

    if !inputs.iter().all(|s| s.dtype().is_numeric()) {
        return Err(PolarsError::ComputeError("All columns must be numeric.".into()));
    }

    let df = df!("status"=>inputs[0].clone(), "time_exit"=>inputs[1].clone())?;
    let table = df.lazy().group_by(["time_exit"]).agg([
        len().alias("cnt")
        , col("status").sum().alias("events")
    ]).sort(["time_exit"], SortMultipleOptions::default()).with_column(
        (lit(n1 as u32) - col("cnt").cum_sum(false).shift_and_fill(1, lit(0))).alias("n_at_risk")
    ).select([
        col("time_exit").alias("t"),
        (lit(1f64) - col("events").cast(DataType::Float64) / col("n_at_risk").cast(DataType::Float64)).cum_prod(false).alias("prob")
    ]).collect()?;

    let ca = table.into_struct("kaplan_meier".into());
    Ok(ca.into_series())
}