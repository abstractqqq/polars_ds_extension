/// KS statistics.
use super::{generic_stats_output, simple_stats_output};
use ordered_float::OrderedFloat;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[inline(always)]
fn binary_search_right(arr: &[OrderedFloat<f64>], t: &OrderedFloat<f64>) -> usize {
    // Can likely get rid of the partial_cmp, because I have gauranteed the values to be finite
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + ((right - left) >> 1);
        match arr[mid].cmp(t) {
            std::cmp::Ordering::Greater => right = mid,
            _ => left = mid + 1,
        }
    }
    left
}

/// Currently only supports two-sided. Won't be too hard to do add one-sided? I hope.
/// Reference:
/// https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_stats_py.py#L8644-L8875
/// Instead of returning a pvalue, the D_n_m quantity is returned, see
/// https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
#[inline]
fn ks_2samp(v1: &[f64], v2: &[f64], alpha: f64) -> (f64, f64) {
    // It is possible to not do binary search because v1 and v2 are already sorted.
    // But that makes the algorithm more complicated.

    let n1: f64 = v1.len() as f64;
    let n2: f64 = v2.len() as f64;

    let v1 = unsafe { std::mem::transmute::<&[f64], &[OrderedFloat<f64>]>(v1) };

    let v2 = unsafe { std::mem::transmute::<&[f64], &[OrderedFloat<f64>]>(v2) };

    // Follow SciPy's trick to compute the difference between two CDFs
    let stats = v1
        .iter()
        .chain(v2.iter())
        .map(|x| {
            (
                (binary_search_right(v1, x) as f64) / n1,
                (binary_search_right(v2, x) as f64) / n2,
            )
        })
        .fold(f64::MIN, |acc, (x, y)| acc.max((x - y).abs()));

    // This differs from SciPy, since I am assuming we are doing two-sided test

    let c_alpha = (-0.5 * (alpha / 2.0).ln()).abs();
    let p_estimate = (c_alpha * (n1 + n2) / (n1 * n2)).sqrt();
    (stats, p_estimate)
}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_ks_2samp(inputs: &[Series]) -> PolarsResult<Series> {
    let s1 = inputs[0].f64()?; // input is sorted, one chunk, gauranteed by Python side code
    let s2 = inputs[1].f64()?; // input is sorted, one chunk, gauranteed by Python side code
    let alpha = inputs[2].f64()?;
    let alpha = alpha.get(0).unwrap();

    if (s1.len() <= 30) || (s2.len() <= 30) {
        generic_stats_output(0f64, f64::NAN)
    } else {
        let v1 = s1.cont_slice().unwrap();
        let v2 = s2.cont_slice().unwrap();
        let (s, p) = ks_2samp(v1, v2, alpha);
        generic_stats_output(s, p)
    }
}
