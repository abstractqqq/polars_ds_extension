/// KS statistics.
use crate::stats_ext::StatsResult;
use crate::utils::binary_search_right;
use itertools::Itertools;
use polars::prelude::*;
use polars::series::IsSorted;
use pyo3_polars::derive::polars_expr;

#[inline]
fn ks_2samp(v1: Vec<f64>, v2: Vec<f64>, stats_only: bool) -> StatsResult {
    // Currently only supports two-sided. Won't be too hard to do add one-sided? I hope.
    // Reference:
    // https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_stats_py.py#L8644-L8875
    // Currently does not support returning p value

    // v1 and v2 must be sorted
    let n1: f64 = v1.len() as f64;
    let n2: f64 = v2.len() as f64;

    // Follow SciPy's trick to compute the difference between two CDFs
    let cdf1_iter = v1
        .iter()
        .chain(v2.iter())
        .map(|x| (binary_search_right(&v1, x).unwrap() as f64) / n1);

    // If we can make binary_search_right work on iterators, we then can work purely on
    // iterators and collect only once
    let cdf2_iter = v1
        .iter()
        .chain(v2.iter())
        .map(|x| (binary_search_right(&v2, x).unwrap() as f64) / n2);

    // This differs from SciPy, since I am assuming we are doing two-sided test
    let diff_iter = cdf1_iter.zip(cdf2_iter).map(|(x, y)| (x - y).abs());

    let stats: f64 = diff_iter.fold(f64::MIN, |acc, x| acc.max(x));
    if stats_only {
        StatsResult::from_stats(stats)
    } else {
        // Temporary
        StatsResult::from_stats(stats)
    }
}

#[polars_expr(output_type=Float64)]
fn pl_ks_2samp(inputs: &[Series]) -> PolarsResult<Series> {
    let s1 = inputs[0].f64()?;
    let s2 = inputs[1].f64()?;

    // Always true right now
    let stats_only = inputs[2].bool()?;
    let stats_only = stats_only.get(0).unwrap();

    // Get rid of these checks? Or how to make them faster? SciPy isn't checking for NaN or Inf..
    let nan_count = s1.is_nan().sum().unwrap() + s2.is_nan().sum().unwrap();
    let inf_count = s1.is_infinite().sum().unwrap() + s2.is_infinite().sum().unwrap();
    let invalid = (nan_count + inf_count) > 0;

    if invalid {
        // Return NaN instead?
        return Err(PolarsError::ComputeError(
            "KS: Input should not contain Inf or NaN.".into(),
        ));
    }

    let v1: Vec<f64> = match s1.is_sorted_flag() {
        IsSorted::Ascending => s1.into_no_null_iter().collect(),
        IsSorted::Descending => s1.into_no_null_iter().rev().collect(),
        _ => s1
            .into_no_null_iter()
            .sorted_unstable_by(|a, b| a.partial_cmp(b).unwrap())
            .collect(),
    };
    let v2: Vec<f64> = match s2.is_sorted_flag() {
        IsSorted::Ascending => s2.into_no_null_iter().collect(),
        IsSorted::Descending => s2.into_no_null_iter().rev().collect(),
        _ => s2
            .into_no_null_iter()
            .sorted_unstable_by(|a, b| a.partial_cmp(b).unwrap())
            .collect(),
    };

    if (v1.len() == 0) | (v2.len() == 0) {
        return Err(PolarsError::ComputeError(
            "KS: Both input series must contain at least 1 non-null values.".into(),
        ));
    }

    let res = ks_2samp(v1, v2, stats_only);
    let s = res.statistic;
    Ok(Series::from_iter([s]))
}
