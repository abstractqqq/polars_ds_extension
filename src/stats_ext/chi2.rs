use super::{simple_stats_output, StatsResult};
use crate::stats::gamma;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_chi2(inputs: &[Series]) -> PolarsResult<Series> {
    let s1_name = inputs[0].name();
    let s2_name = inputs[1].name();

    let u1 = inputs[0].unique()?;
    let u1_len = u1.len();
    let u2 = inputs[1].unique()?;
    let u2_len = u2.len();
    // Get the cartesian product
    let df1 = df!(s1_name => u1)?.lazy();
    let df2 = df!(s2_name => u2)?.lazy();
    let cross = df1.cross_join(df2);

    // Create a "fake" contigency table
    let df3 = DataFrame::from_iter(inputs[0..2].to_vec())
        .lazy()
        .group_by([col(s1_name), col(s2_name)])
        .agg([count().alias("ob")]);

    let df4 = cross
        .join(
            df3,
            [col(s1_name), col(s2_name)],
            [col(s1_name), col(s2_name)],
            JoinArgs::new(JoinType::Left),
        )
        .with_column(col("ob").fill_null(0));

    // Compute the statistic
    let mut final_df = df4
        .with_columns([
            ((col("ob").sum().over([s2_name]) * col("ob").sum().over([s1_name]))
                .cast(DataType::Float64)
                / col("ob").sum().cast(DataType::Float64))
            .alias("ex"),
        ])
        .select([
            ((col("ob").cast(DataType::Float64) - col("ex")).pow(2) / col("ex"))
                .sum()
                .alias("output"),
        ])
        .collect()?;

    // Get the statistic
    let out = final_df.drop_in_place("output")?;
    let stats = out.f64()?;
    let stats = stats.get(0).unwrap();
    // Compute p value. It is a special case of Gamma distribution
    let dof = u1_len.abs_diff(1) * u2_len.abs_diff(1);
    let (shape, rate) = (dof as f64 / 2., 0.5);
    let p = gamma::sf(stats, shape, rate).map_err(|e| PolarsError::ComputeError(e.into()));
    let p = p?;
    // Get output
    let s = Series::from_vec("statistic", vec![stats]);
    let p = Series::from_vec("pvalue", vec![p]);
    let out = StructChunked::new("", &[s, p])?;

    Ok(out.into_series())
}
