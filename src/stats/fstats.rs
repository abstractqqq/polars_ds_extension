/// Multiple F-statistics at once and F test
use super::{simple_stats_output, StatsResult};
use crate::{stats_utils::beta::fisher_snedecor_sf, utils::list_f64_output};
use itertools::Itertools;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[inline]
fn ftest(x: f64, f1: f64, f2: f64) -> Result<StatsResult, String> {
    if x < 0. {
        return Err("F statistics is < 0. This should be impossible.".into());
    }
    // F test does not take alternative.
    let p = fisher_snedecor_sf(x, f1, f2)?;
    Ok(StatsResult::new(x, p))
}

/// An internal helper function to compute f statistic for F test, with the option to comput
/// the p value too. It shouldn't be used outside. The API is bad for outsiders to use.
/// When return_p is false, returns a Vec with f stats.
/// When return_p is true, returns a Vec that has x_0, .., x_{n-1} = f_0, .., f_{n-1}
/// where n = inputs.len() - 1 = number of features
/// And additionally x_n, .., x_{2n - 2} = p_0, .., p_{n-1}, are the p values.
///
/// Refactor?
fn _f_stats(inputs: &[Series], return_p: bool) -> PolarsResult<Vec<f64>> {
    let target = "target";
    let v = inputs
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            if i == 0 {
                s.clone().with_name(target)
            } else {
                s.clone().with_name(i.to_string().as_str())
            }
        })
        .collect_vec();
    let n_cols = v.len();

    let df = DataFrame::new(v)?.lazy();
    // inputs[0] is the group
    // all the rest should numerical
    let mut step_one: Vec<Expr> = Vec::with_capacity(inputs.len() * 2 - 1);
    step_one.push(len().cast(DataType::Float64).alias("cnt"));
    let mut step_two: Vec<Expr> = Vec::with_capacity(inputs.len() + 1);
    step_two.push(col("cnt").sum().alias("n_samples").cast(DataType::UInt32));
    step_two.push(
        col(target)
            .count()
            .alias("n_classes")
            .cast(DataType::UInt32),
    );

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
        .group_by([col(target)])
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
            "Number of classes, or number of samples is either 1 or 0, which is invalid.".into(),
        ));
    }

    let df_btw_class = n_classes.abs_diff(1) as f64;
    let df_in_class = n_samples.abs_diff(n_classes) as f64;

    // fstats is 2D
    let fstats = reference.to_ndarray::<Float64Type>(IndexOrder::default())?;
    let scale = df_in_class / df_btw_class;

    let out: Vec<f64> = if return_p {
        let mut output: Vec<f64> = Vec::with_capacity(reference.height() * 2);
        for f in fstats.row(0) {
            output.push(f * scale);
        }
        for f in fstats.row(0) {
            let res = ftest(f * scale, df_btw_class, df_in_class);
            match res {
                Ok(s) => {
                    if let Some(p) = s.p {
                        output.push(p);
                    } else {
                        return Err(PolarsError::ComputeError(
                            "F test: Unknown error occurred when computing P value.".into(),
                        ));
                    }
                }
                Err(e) => return Err(PolarsError::ComputeError(e.into())),
            }
        }
        output
    } else {
        fstats.row(0).into_iter().map(|f| f * scale).collect_vec()
    };

    Ok(out)
}

/// Use inputs[0] as the grouping column
/// and inputs[1..] as other columns. Compute F statistic for other columns w.r.t the grouping column.
/// Outputs a list of floats, in the order of other columns.
#[polars_expr(output_type_func=list_f64_output)]
fn pl_f_stats(inputs: &[Series]) -> PolarsResult<Series> {
    let stats = _f_stats(inputs, false)?;
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        "f_stats",
        1,
        stats.len(),
        DataType::Float64,
    );

    builder.append_slice(stats.as_slice());
    let output = builder.finish();
    Ok(output.into_series())
}

/// Use inputs[0] as the grouping column
/// and inputs[1] as the column to run F-test. There should be only two columns.
#[polars_expr(output_type_func=simple_stats_output)]
fn pl_f_test(inputs: &[Series]) -> PolarsResult<Series> {
    // The variable res has 2 values, the test statistic and p value.
    let res = _f_stats(&inputs[..2], true)?;
    let s = Series::from_vec("statistic", vec![res[0]]);
    let p = Series::from_vec("pvalue", vec![res[1]]);
    let out = StructChunked::new("", &[s, p])?;
    Ok(out.into_series())
}
