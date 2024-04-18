use super::simple_stats_output;
use crate::stats_utils::normal;
use polars::{prelude::*, series::ops::NullBehavior};
use pyo3_polars::derive::polars_expr;

fn _xi_corr(inputs: &[Series]) -> PolarsResult<Series> {
    // Input 0 should be x.rank(method="random")
    // Input 1 should be y.rank(method="max").cast(pl.Float64).alias("r")
    // Input 2 should be (-y).rank(method="max").cast(pl.Float64).alias("l")

    let df = df!("x_rk" => &inputs[0], "r" => &inputs[1], "l" => &inputs[2])?.lazy();
    Ok(df
        .sort(["x_rk"], Default::default())
        .select([(lit(1.0)
            - ((len().cast(DataType::Float64) / lit(2.0))
                * col("r").diff(1, NullBehavior::Ignore).abs().sum())
                / (col("l") * (len() - col("l"))).sum())
        .alias("statistic")])
        .collect()?
        .drop_in_place("statistic")
        .unwrap())
}

#[polars_expr(output_type=Float64)]
pub fn pl_xi_corr(inputs: &[Series]) -> PolarsResult<Series> {
    _xi_corr(inputs)
}

#[polars_expr(output_type_func=simple_stats_output)]
pub fn pl_xi_corr_w_p(inputs: &[Series]) -> PolarsResult<Series> {
    let n = inputs[0].len();
    let corr = _xi_corr(inputs)?;
    let p: f64 = if n < 30 {
        f64::NAN
    } else {
        let sqrt_n = (n as f64).sqrt();
        let c = corr.f64().unwrap();
        let c = c.get(0).unwrap();
        // Two sided
        normal::sf_unchecked(sqrt_n * c.abs() / (0.4f64).sqrt(), 0., 1.0) * 2.0
    };
    let p = Series::from_vec("pvalue", vec![p]);
    let out = StructChunked::new("xi_corr", &[corr, p])?;
    Ok(out.into_series())
}
