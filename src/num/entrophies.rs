use super::l_inf_dist;
use crate::num::knn::{build_knn_matrix_data, build_standard_kdtree, query_nb_cnt, KdtreeKwargs};
use ndarray::s;
use polars::prelude::*;
use polars_core::POOL;
use pyo3_polars::derive::{polars_expr, CallerContext};
use pyo3_polars::export::polars_core::utils::rayon::iter::{
    IntoParallelIterator, ParallelIterator,
};

// https://en.wikipedia.org/wiki/Sample_entropy
// https://en.wikipedia.org/wiki/Approximate_entropy

// Could be made faster once https://github.com/mrhooray/kdtree-rs/pull/52 is merged

#[polars_expr(output_type=Float64)]
fn pl_approximate_entropy(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeKwargs,
) -> PolarsResult<Series> {
    // inputs[0] is radius, the rest are the shifted columns
    // Set up radius. r is a scalar and set up at Python side.

    let radius = inputs[0].f64()?;
    let name = inputs[1].name();
    if radius.get(0).is_none() {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let r = radius.get(0).unwrap();
    let dim = inputs[1..].len();

    let data = build_knn_matrix_data(&inputs[1..])?;
    let n1 = data.nrows(); // This is equal to original length - m + 1
                           // Here, dim equals to run_length + 1, or m + 1
                           // + 1 because I am intentionally generating one more, so that we do to_ndarray only once.
    if (n1 < dim) || (r <= 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let can_parallel = kwargs.parallel && !context.parallel();
    let leaf_size = kwargs.leaf_size;

    // Step 3, 4, 5 in wiki
    let data_1_view = data.slice(s![..n1, ..dim.abs_diff(1)]);
    let tree = build_standard_kdtree(dim.abs_diff(1), leaf_size, &data_1_view, None)?;
    let nb_in_radius = query_nb_cnt(&tree, data_1_view, &l_inf_dist, r, can_parallel);
    let phi_m: f64 = nb_in_radius
        .into_no_null_iter()
        .fold(0_f64, |acc, x| acc + (x as f64 / n1 as f64).ln())
        / n1 as f64;

    // Step 3, 4, 5 for m + 1 in wiki
    let n2 = n1.abs_diff(1);
    let data_2_view = data.slice(s![..n2, ..]);
    let tree = build_standard_kdtree(dim, leaf_size, &data_2_view, None)?;
    let nb_in_radius = query_nb_cnt(&tree, data_2_view, &l_inf_dist, r, can_parallel);
    let phi_m1: f64 = nb_in_radius
        .into_no_null_iter()
        .fold(0_f64, |acc, x| acc + (x as f64 / n2 as f64).ln())
        / n2 as f64;

    // Output
    Ok(Series::from_vec(name, vec![(phi_m1 - phi_m).abs()]))
}

#[polars_expr(output_type=Float64)]
fn pl_sample_entropy(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeKwargs,
) -> PolarsResult<Series> {
    // inputs[0] is radius, the rest are the shifted columns
    // Set up radius. r is a scalar and set up at Python side.
    let radius = inputs[0].f64()?;
    let name = inputs[1].name();
    if radius.get(0).is_none() {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let r = radius.get(0).unwrap();
    let dim = inputs[1..].len();
    let data = build_knn_matrix_data(&inputs[1..])?;
    let n1 = data.nrows(); // This is equal to original length - m + 1
                           // Here, dim equals to run_length + 1, or m + 1
                           // + 1 because I am intentionally generating one more, so that we do to_ndarray only once.
    if (n1 < dim) || (r <= 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    let leaf_size = kwargs.leaf_size;

    let data_1_view = data.slice(s![..n1, ..dim.abs_diff(1)]);
    let tree = build_standard_kdtree(dim.abs_diff(1), leaf_size, &data_1_view, None)?;
    let nb_in_radius = query_nb_cnt(&tree, data_1_view, &l_inf_dist, r, can_parallel);
    let b = (nb_in_radius.sum().unwrap_or(0) as f64) - (n1 as f64);

    let n2 = n1.abs_diff(1);
    let data_2_view = data.slice(s![..n2, ..]);
    let tree = build_standard_kdtree(dim, leaf_size, &data_2_view, None)?;
    let nb_in_radius = query_nb_cnt(&tree, data_2_view, &l_inf_dist, r, can_parallel);
    let a = (nb_in_radius.sum().unwrap_or(0) as f64) - (n2 as f64);

    // Output
    Ok(Series::from_vec(name, vec![(b / a).ln()]))
}

#[polars_expr(output_type=Float64)]
fn pl_knn_entropy(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeKwargs,
) -> PolarsResult<Series> {
    // Define inputs
    let can_parallel = kwargs.parallel && !context.parallel();
    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;

    let name = inputs[0].name();
    let dim = inputs.len();

    let data = build_knn_matrix_data(inputs)?;
    let nrows = data.nrows();

    if nrows <= k {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }

    // Get cd
    let metric_str = kwargs.metric.as_str();
    let n = nrows as f64;
    let d = dim as f64;
    // Should support l1, l2, inf here.

    let (dist_func, cd): (fn(&[f64], &[f64]) -> f64, f64) = if metric_str == "l2" {
        let half_d: f64 = d / 2.0;
        let cd = std::f64::consts::PI.powf(half_d) / (2f64.powf(d)) / (1.0 + half_d).gamma();
        (super::l2_dist, cd) // Need l2 with square root
    } else if metric_str == "inf" {
        (super::l_inf_dist, 1.0)
    } else {
        return Err(PolarsError::ComputeError(
            "Distance metric not implemented.".into(),
        ));
    };

    // G1
    let g1 = crate::stats_utils::gamma::digamma(n) - crate::stats_utils::gamma::digamma(k as f64);

    // KNN part
    let data_view = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &data_view, None)?;

    let logd = if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = crate::utils::split_offsets(nrows, n_threads);
        let partial_sums = splits.into_par_iter().map(|(offset, len)| {
            let piece = data.slice(s![offset..offset + len, 0..dim]);
            let mut out: f64 = 0.0;
            for p in piece.rows() {
                let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
                if let Ok(mut v) = tree.nearest(s, k + 1, &dist_func) {
                    let (d, _) = v.pop().unwrap();
                    out += (2.0 * d).ln();
                }
            }
            out
        });
        POOL.install(|| partial_sums.sum())
    } else {
        let mut out: f64 = 0.0;
        for p in data.rows() {
            let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(mut v) = tree.nearest(s, k + 1, &dist_func) {
                // The last element represents the k-th neighbor
                // The pop should be safe, because nrows > k
                let (d, _) = v.pop().unwrap();
                out += (2.0 * d).ln();
            }
        }
        out
    };

    let out = g1 + cd.ln() + logd * d / n;
    let ca = Float64Chunked::from_slice(name, &[out]);
    Ok(ca.into_series())
}
