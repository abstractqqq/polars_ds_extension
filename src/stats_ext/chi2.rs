use super::{generic_stats_output, simple_stats_output};
use crate::stats_utils::gamma;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn chi2_full_output(fields: &[Field]) -> PolarsResult<Field> {
    let s = Field::new("statistic".into(), DataType::Float64);
    let p = Field::new("pvalue".into(), DataType::Float64);
    let dof = Field::new("dof".into(), DataType::UInt32);
    let f1 = fields[0].clone();
    let f2 = fields[1].clone();
    let ef = Field::new("E[freq]".into(), DataType::Float64);
    let v: Vec<Field> = vec![s, p, dof, f1, f2, ef];
    Ok(Field::new("chi2_full".into(), DataType::Struct(v)))
}

fn _chi2_helper(inputs: &[Series]) -> PolarsResult<(LazyFrame, usize, usize)> {
    // Return a df with necessary values to compute chi2, together
    // with nrows and ncols
    let s1_name = "s1";
    let s2_name = "s2";

    let u1 = inputs[0].unique()?;
    let u1_len = u1.len();
    let u2 = inputs[1].unique()?;
    let u2_len = u2.len();
    // Get the cartesian product
    let df1 = df!(s1_name => u1)?.lazy();
    let df2 = df!(s2_name => u2)?.lazy();
    let cross = df1.cross_join(df2, None);

    // Create a "fake" contigency table
    let s1 = inputs[0].clone();
    let s2 = inputs[1].clone();
    let df3 = df!(s1_name => s1, s2_name => s2)?
        .lazy()
        .group_by([col(s1_name), col(s2_name)])
        .agg([len().cast(DataType::UInt64).alias("ob")]);

    let df4 = cross
        .join(
            df3,
            [col(s1_name), col(s2_name)],
            [col(s1_name), col(s2_name)],
            JoinArgs::new(JoinType::Left),
        )
        .with_column(col("ob").fill_null(0));

    // Compute the statistic
    let frame = df4.with_columns([((col("ob").sum().over([s2_name])
        * col("ob").sum().over([s1_name]))
    .cast(DataType::Float64)
        / col("ob").sum().cast(DataType::Float64))
    .alias("ex")]);

    Ok((frame, u1_len, u2_len))
}

fn _chi2_pvalue(stats: f64, dof: usize) -> PolarsResult<f64> {
    // The p value for chi2
    let p = if stats.is_nan() {
        f64::NAN
    } else {
        let (shape, rate) = (dof as f64 / 2., 0.5);
        let p = gamma::sf(stats, shape, rate).map_err(|e| PolarsError::ComputeError(e.into()));
        p?
    };
    Ok(p)
}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_chi2(inputs: &[Series]) -> PolarsResult<Series> {
    let (df, u1_len, u2_len) = _chi2_helper(inputs)?;

    let mut final_df = df
        .select([
            ((col("ob").cast(DataType::Float64) - col("ex")).pow(2) / col("ex"))
                .sum()
                .alias("output"),
        ])
        .collect()?;

    // Get the statistic
    let out = final_df.drop_in_place("output").unwrap();
    let stats = out.f64()?;
    let stats = stats.get(0).unwrap_or(f64::NAN);
    // Compute p value. It is a special case of Gamma distribution
    let dof = u1_len.abs_diff(1) * u2_len.abs_diff(1);
    let p = _chi2_pvalue(stats, dof)?;
    generic_stats_output(stats, p)
}

#[polars_expr(output_type_func=chi2_full_output)]
fn pl_chi2_full(inputs: &[Series]) -> PolarsResult<Series> {
    let s1_name = inputs[0].name();
    let s2_name = inputs[1].name();

    let (df, u1_len, u2_len) = _chi2_helper(inputs)?;
    // cheap clone
    let mut df2 = df
        .clone()
        .select([
            col("s1").alias(s1_name.clone()),
            col("s2").alias(s2_name.clone()),
            col("ex").alias("E[freq]"),
        ])
        .collect()?;
    let ef = df2.drop_in_place("E[freq]").unwrap();
    let s1 = df2.drop_in_place(s1_name).unwrap();
    let s2 = df2.drop_in_place(s2_name).unwrap();

    let mut final_df = df
        .select([
            ((col("ob").cast(DataType::Float64) - col("ex")).pow(2) / col("ex"))
                .sum()
                .alias("output"),
        ])
        .collect()?;

    // Get the statistic
    let out = final_df.drop_in_place("output").unwrap();
    let stats = out.f64()?;
    let stats = stats.get(0).unwrap_or(f64::NAN);
    let dof = u1_len.abs_diff(1) * u2_len.abs_diff(1);
    let p = _chi2_pvalue(stats, dof)?;
    let stats_column = Column::new_scalar("statistic".into(), stats.into(), 1);
    let pval_column = Column::new_scalar("pvalue".into(), p.into(), 1);
    let dof_column = Column::new_scalar("dof".into(), (dof as u32).into(), 1);

    let ca = StructChunked::from_columns(
        "chi2_full".into(),
        ef.len(),
        &[stats_column, pval_column, dof_column, s1, s2, ef],
    )?;
    Ok(ca.into_series())
}
