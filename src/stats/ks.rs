/// KS statistics.
use crate::stats::StatsResult;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[inline(always)]
fn binary_search_right<T: PartialOrd>(arr: &[T], t: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + ((right - left) >> 1);
        if let Some(c) = arr[mid].partial_cmp(t) {
            match c {
                std::cmp::Ordering::Greater => right = mid,
                _ => left = mid + 1,
            }
        } else {
            return None;
        }
    }
    Some(left)
}

#[inline]
fn ks_2samp(v1: &[f64], v2: &[f64], stats_only: bool) -> StatsResult {
    // Currently only supports two-sided. Won't be too hard to do add one-sided? I hope.
    // Reference:
    // https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_stats_py.py#L8644-L8875
    // Currently does not support returning p value

    // v1 and v2 must be sorted
    let n1: f64 = v1.len() as f64;
    let n2: f64 = v2.len() as f64;

    // Follow SciPy's trick to compute the difference between two CDFs
    let stats = v1
        .iter()
        .chain(v2.iter())
        .map(|x| {
            (
                (binary_search_right(&v1, x).unwrap() as f64) / n1,
                (binary_search_right(&v2, x).unwrap() as f64) / n2,
            )
        })
        .fold(f64::MIN, |acc, (x, y)| acc.max((x - y).abs()));

    // This differs from SciPy, since I am assuming we are doing two-sided test

    if stats_only {
        StatsResult::from_stats(stats)
    } else {
        // Temporary
        StatsResult::from_stats(stats)
    }
}

#[polars_expr(output_type=Float64)]
fn pl_ks_2samp(inputs: &[Series]) -> PolarsResult<Series> {
    let s1 = inputs[0].f64()?; // input is sorted, one chunk, gauranteed by Python side code
    let s2 = inputs[1].f64()?; // input is sorted, one chunk, gauranteed by Python side code

    if (s1.len() == 0) || (s2.len() == 0) {
        return Err(PolarsError::ComputeError(
            "KS: Both input series must contain at least 1 non-null values.".into(),
        ));
    }
    // Always true right now
    let stats_only = inputs[2].bool()?;
    let stats_only = stats_only.get(0).unwrap_or(true);

    let v1 = s1.cont_slice().unwrap();
    let v2 = s2.cont_slice().unwrap();

    let res = ks_2samp(v1, v2, stats_only);
    let s = res.statistic;
    Ok(Series::from_iter([s]))
}
