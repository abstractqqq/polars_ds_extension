use crate::arkadia::{arkadia_any::AnyKDT, matrix_to_empty_leaves, SpacialQueries};
use crate::num_ext::knn::{query_nb_cnt, KDTKwargs};
use crate::utils::{series_to_ndarray, split_offsets, DIST};
use ndarray::{s, ArrayView2};
use polars::prelude::*;
use polars_core::POOL;
use pyo3_polars::derive::{polars_expr, CallerContext};
use pyo3_polars::export::polars_core::utils::rayon::iter::{
    IntoParallelIterator, ParallelIterator,
};

// https://en.wikipedia.org/wiki/Sample_entropy
// https://en.wikipedia.org/wiki/Approximate_entropy

#[polars_expr(output_type=Float64)]
fn pl_approximate_entropy(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTKwargs,
) -> PolarsResult<Series> {
    // inputs[0] is radius, the rest are the shifted columns
    // Set up radius. r is a scalar and set up at Python side.

    let name = inputs[1].name();
    let data = series_to_ndarray(&inputs[1..], IndexOrder::C)?;
    let radius = inputs[0].f64()?;
    if radius.len() != 1 {
        return Err(PolarsError::ComputeError("Radius must be a scalar.".into()));
    }

    let r = radius.get(0).unwrap();
    let dim = inputs[1..].len();
    let n1 = data.nrows(); // This is equal to original length - m + 1
                           // Here, dim equals to run_length + 1, or m + 1
                           // + 1 because I am intentionally generating one more, so that we do to_ndarray only once.
    if (n1 < dim) || (r <= 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let can_parallel = kwargs.parallel && !context.parallel();

    // Step 3, 4, 5 in wiki
    let data_1_view = data.slice(s![..n1, ..dim.abs_diff(1)]);
    let mut leaves = matrix_to_empty_leaves(&data_1_view);
    let tree = AnyKDT::from_leaves_unchecked(&mut leaves, DIST::LINF);

    let nb_in_radius = query_nb_cnt(tree, data_1_view, r, can_parallel);
    let phi_m: f64 = nb_in_radius
        .into_no_null_iter()
        .fold(0_f64, |acc, x| acc + (x as f64 / n1 as f64).ln())
        / n1 as f64;

    // Step 3, 4, 5 for m + 1 in wiki
    let n2 = n1.abs_diff(1);
    let data_2_view = data.slice(s![..n2, ..]);
    let mut leaves2 = matrix_to_empty_leaves(&data_2_view);
    let tree = AnyKDT::from_leaves_unchecked(&mut leaves2, DIST::LINF);

    let nb_in_radius = query_nb_cnt(tree, data_2_view, r, can_parallel);
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
    kwargs: KDTKwargs,
) -> PolarsResult<Series> {
    // inputs[0] is radius, the rest are the shifted columns
    // Set up radius. r is a scalar and set up at Python side.
    let radius = inputs[0].f64()?;
    if radius.len() != 1 {
        return Err(PolarsError::ComputeError("Radius must be a scalar.".into()));
    }

    let r = radius.get(0).unwrap_or(-1f64); // see return below
    let name = inputs[1].name();
    let dim = inputs[1..].len();
    let data = series_to_ndarray(&inputs[1..], IndexOrder::C)?;
    let n1 = data.nrows();
    // This is equal to original length - m + 1
    // Here, dim equals to run_length + 1, or m + 1
    // + 1 because I am intentionally generating one more, so that we do to_ndarray only once.
    if (n1 < dim) || (r <= 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();

    let data_1_view = data.slice(s![..n1, ..dim.abs_diff(1)]);
    let mut leaves = matrix_to_empty_leaves(&data_1_view);
    let tree = AnyKDT::from_leaves_unchecked(&mut leaves, DIST::LINF);

    let nb_in_radius = query_nb_cnt(tree, data_1_view, r, can_parallel);
    let b = (nb_in_radius.sum().unwrap_or(0) as f64) - (n1 as f64);

    let n2 = n1.abs_diff(1);
    let data_2_view = data.slice(s![..n2, ..]);
    let mut leaves2 = matrix_to_empty_leaves(&data_2_view);
    let tree = AnyKDT::from_leaves_unchecked(&mut leaves2, DIST::LINF);

    let nb_in_radius = query_nb_cnt(tree, data_2_view, r, can_parallel);
    let a = (nb_in_radius.sum().unwrap_or(0) as f64) - (n2 as f64);

    // Output
    Ok(Series::from_vec(name, vec![(b / a).ln()]))
}

/// Comptues the logd part of the KNN entropy
fn _knn_entropy_helper<'a>(
    tree: AnyKDT<'a, f64, ()>,
    data: ArrayView2<f64>,
    k: usize,
    can_parallel: bool,
) -> f64 {
    if can_parallel {
        let splits = split_offsets(data.nrows(), POOL.current_num_threads());
        let partial_sums = splits.into_par_iter().map(|(offset, len)| {
            let piece = data.slice(s![offset..offset + len, ..]);
            piece.rows().into_iter().fold(0f64, |acc, row| {
                if let Some(mut v) = tree.knn(k + 1, row.as_slice().unwrap(), 0.) {
                    let nb = v.pop().unwrap();
                    acc + (2.0 * nb.to_dist()).ln()
                } else {
                    acc
                }
            })
        });
        POOL.install(|| partial_sums.sum())
    } else {
        data.rows().into_iter().fold(0f64, |acc, row| {
            if let Some(mut v) = tree.knn(k + 1, row.as_slice().unwrap(), 0.) {
                let nb = v.pop().unwrap();
                acc + (2.0 * nb.to_dist()).ln()
            } else {
                acc
            }
        })
    }
}

#[polars_expr(output_type=Float64)]
fn pl_knn_entropy(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTKwargs,
) -> PolarsResult<Series> {
    // Define inputs
    let can_parallel = kwargs.parallel && !context.parallel();
    let k = kwargs.k;

    let name = inputs[0].name();
    let dim = inputs.len();

    let data = series_to_ndarray(inputs, IndexOrder::C)?;
    let nrows = data.nrows();

    if nrows <= k {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }

    let metric_str = kwargs.metric.as_str();
    let n = nrows as f64;
    let d = dim as f64;

    // G1
    let g1 = crate::stats_utils::gamma::digamma(n) - crate::stats_utils::gamma::digamma(k as f64);

    // Should support l1, l2, inf here.

    let data_view = data.view();
    let (cd, log_d) = if metric_str == "l2" {
        let half_d: f64 = d / 2.0;
        let cd = std::f64::consts::PI.powf(half_d) / (2f64.powf(d)) / (1.0 + half_d).gamma();
        let mut leaves = matrix_to_empty_leaves(&data_view);
        let tree = AnyKDT::from_leaves_unchecked(&mut leaves, DIST::L2);

        (cd, _knn_entropy_helper(tree, data_view, k, can_parallel))
    } else if metric_str == "inf" {
        let cd = 1.0;
        let mut leaves = matrix_to_empty_leaves(&data_view);
        let tree = AnyKDT::from_leaves_unchecked(&mut leaves, DIST::LINF);

        (cd, _knn_entropy_helper(tree, data_view, k, can_parallel))
    } else {
        return Err(PolarsError::ComputeError(
            "KNN Entropy for distance metric is  not implemented.".into(),
        ));
    };

    let out = g1 + cd.ln() + log_d * d / n;
    let ca = Float64Chunked::from_slice(name, &[out]);
    Ok(ca.into_series())
}
