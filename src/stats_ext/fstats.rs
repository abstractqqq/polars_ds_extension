/// Multiple F-statistics at once and F test
use super::simple_stats_output;
use crate::stats_utils::beta::fisher_snedecor_sf;
use crate::utils::{columns_to_vec, IndexOrder};
use core::f64;
use itertools::Itertools;
use polars::frame::column::Column;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// Use inputs[0] as the target column (discrete, indicating the groups)
/// and inputs[i] as the column to run F-test against the target, i > 0.
#[polars_expr(output_type_func=simple_stats_output)]
fn pl_f_test(inputs: &[Series]) -> PolarsResult<Series> {
    // Use a df to make the computations parallel.
    // Column at index 0 is the target column
    let v = inputs
        .into_iter()
        .enumerate()
        .map(|(i, s)| Column::new(i.to_string().into(), s))
        .collect_vec();
    let n_cols = v.len();

    let df = DataFrame::new(v)?.lazy();
    // inputs[0] is the group
    // all the rest should numerical
    let mut step_one: Vec<Expr> = Vec::with_capacity(inputs.len() * 2 - 1);
    step_one.push(len().cast(DataType::Float64).alias("cnt"));
    let mut step_two: Vec<Expr> = Vec::with_capacity(inputs.len() + 1);
    step_two.push(col("cnt").sum().cast(DataType::UInt32).alias("n_samples"));
    step_two.push(col("0").count().cast(DataType::UInt32).alias("n_classes"));

    for i in 1..n_cols {
        let name = i.to_string();
        let name = name.as_str();
        let n_sum = format!("{}_sum", i);
        let n_sum = n_sum.as_str();
        let n_var = format!("{}_var", i);
        let n_var = n_var.as_str();
        step_one.push(col(name).sum().alias(n_sum));
        step_one.push(col(name).var(0).alias(n_var));
        let p1: Expr = (col(n_sum).cast(DataType::Float64) / col("cnt").cast(DataType::Float64)
            - (col(n_sum).sum().cast(DataType::Float64)
                / col("cnt").sum().cast(DataType::Float64)))
        .pow(2);
        let p2 = col(n_var).dot(col("cnt").cast(DataType::Float64));

        step_two.push(p1.dot(col("cnt").cast(DataType::Float64)) / p2)
    }

    let mut reference = df
        .group_by([col("0")])
        .agg(step_one)
        .select(step_two)
        .collect()?;

    let n_samples = reference.drop_in_place("n_samples").unwrap();
    let n_classes = reference.drop_in_place("n_classes").unwrap();
    let n_samples = n_samples.u32()?;
    let n_classes = n_classes.u32()?;
    let n_samples = n_samples.get(0).unwrap_or(0);
    let n_classes = n_classes.get(0).unwrap_or(0);

    if n_classes <= 1 || n_samples <= 1 {
        return Err(PolarsError::ComputeError(
            "F-stats: n_classes <= 1 in target or n_samples <= 1.".into(),
        ));
    }

    let df_btw_class = n_classes.abs_diff(1) as f64;
    let df_in_class = n_samples.abs_diff(n_classes) as f64;
    // Note: reference is a df with 1 row. We need to get the stats out
    // fstats is 2D but with 1 row.

    let scale = df_in_class / df_btw_class;

    let mut fstats = columns_to_vec::<Float64Type>(reference.take_columns(), IndexOrder::C)?;
    fstats.iter_mut().for_each(|v| *v = *v * scale);

    let out_p: Vec<f64> = fstats
        .iter()
        .map(|x| fisher_snedecor_sf(*x, df_btw_class, df_in_class).unwrap_or(f64::NAN))
        .collect();

    let s1 = Column::new("statistic".into(), fstats);
    let s2 = Column::new("pvalue".into(), out_p);

    let ca = StructChunked::from_columns("f-test".into(), s1.len(), &[s1, s2])?;
    Ok(ca.into_series())
}
