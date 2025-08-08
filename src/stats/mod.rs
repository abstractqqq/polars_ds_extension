mod chi2;
mod fstats;
mod kendall_tau;
mod ks;
mod mann_whitney_u;
mod normal_test;
mod sample;
mod t_test;
mod xi_corr;

use polars::prelude::*;

fn stats_output_dtype() -> DataType {
    let s = Field::new("statistic".into(), DataType::Float64);
    let p = Field::new("pvalue".into(), DataType::Float64);
    let v: Vec<Field> = vec![s, p];
    DataType::Struct(v)
}

pub fn simple_stats_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new("".into(), stats_output_dtype()))
}

pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

impl From<&str> for Alternative {
    fn from(s: &str) -> Alternative {
        match s.to_lowercase().as_str() {
            "two-sided" | "two" => Alternative::TwoSided,
            "less" => Alternative::Less,
            "greater" => Alternative::Greater,
            _ => Alternative::TwoSided,
        }
    }
}

#[inline]
fn generic_stats_output(statistic: f64, pvalue: f64) -> PolarsResult<Series> {
    let s = Series::from_vec("statistic".into(), vec![statistic]);
    let p = Series::from_vec("pvalue".into(), vec![pvalue]);
    let out = StructChunked::from_series("".into(), 1, [&s, &p].into_iter())?;
    Ok(out.into_series())
}

// #[inline]
// fn generic_optional_stats_output(output: Option<(f64, f64)>) -> PolarsResult<Series> {
//     if let Some((statistic, pvalue)) = output {
//         let s = Series::from_vec("statistic".into(), vec![statistic]);
//         let p = Series::from_vec("pvalue".into(), vec![pvalue]);
//         let out = StructChunked::from_series("".into(), 1, [&s, &p].into_iter())?;
//         Ok(out.into_series())
//     } else {
//         let dtype = stats_output_dtype();
//         Ok(Series::full_null("".into(), 1, &dtype).into_series())
//     }
// }
