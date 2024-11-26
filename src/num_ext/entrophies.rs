use crate::arkadia::utils::slice_to_empty_leaves;
use crate::arkadia::{kdt::KDT, SpatialQueries};
use crate::num_ext::knn::KDTKwargs;
use crate::utils::{series_to_row_major_slice, split_offsets, DIST};
use core::f64;
use polars::prelude::*;
use polars_core::POOL;
use pyo3_polars::derive::{polars_expr, CallerContext};
use pyo3_polars::export::polars_core::{
    error::PolarsError,
    utils::rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator},
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

    let radius = inputs[0].f64()?;
    if radius.len() != 1 {
        return Err(PolarsError::ComputeError("Radius must be a scalar.".into()));
    }
    let r = radius.get(0).unwrap();
    let name = inputs[1].name();
    let ncols = inputs[1..].len();
    let data = series_to_row_major_slice::<Float64Type>(&inputs[1..])?;
    let nrows = data.len() / ncols;

    if (nrows < ncols) || (r <= 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec(name.clone(), vec![f64::NAN]));
    }
    let can_parallel = kwargs.parallel && !context.parallel();

    // Step 3, 4, 5 in wiki

    let ncols_minus_1 = ncols.abs_diff(1);
    let mut leaves = data
        .chunks_exact(ncols)
        .map(|sl| ((), &sl[..ncols_minus_1]).into())
        .collect::<Vec<_>>();

    let tree = KDT::from_leaves_unchecked(&mut leaves, DIST::LINF);

    let nrows_recip = (nrows as f64).recip();
    let phi_m = if can_parallel {
        data.chunks_exact(ncols)
            .par_bridge()
            .map(|sl| {
                (tree.within_count(&sl[..ncols_minus_1], r).unwrap_or(0) as f64 * nrows_recip).ln()
            })
            .sum::<f64>()
            * nrows_recip
    } else {
        data.chunks_exact(ncols).fold(0f64, |acc, sl| {
            acc + (tree.within_count(&sl[..ncols_minus_1], r).unwrap_or(0) as f64 * nrows_recip)
                .ln()
        }) * nrows_recip
    };

    // Step 3, 4, 5 for m + 1 in wiki
    let nrows_minus_1 = nrows.abs_diff(1);

    drop(tree);

    let mut leaves2 = data
        .chunks_exact(ncols)
        .take(nrows_minus_1)
        .map(|sl| ((), sl).into())
        .collect::<Vec<_>>();

    let tree = KDT::from_leaves_unchecked(&mut leaves2, DIST::LINF);

    let nrows_minus_1_recip = 1.0 / (nrows_minus_1 as f64);
    let phi_m1 = if can_parallel {
        data.chunks_exact(ncols)
            .take(nrows_minus_1)
            .par_bridge()
            .map(|sl| (tree.within_count(sl, r).unwrap_or(0) as f64 * nrows_minus_1_recip).ln())
            .sum::<f64>()
            * nrows_minus_1_recip
    } else {
        data.chunks_exact(ncols)
            .take(nrows_minus_1)
            .fold(0f64, |acc, sl| {
                acc + (tree.within_count(sl, r).unwrap_or(0) as f64 * nrows_minus_1_recip).ln()
            })
            * nrows_minus_1_recip
    };
    // Output
    Ok(Series::from_vec(name.clone(), vec![(phi_m1 - phi_m).abs()]))
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
    let ncols = inputs[1..].len();
    let data = series_to_row_major_slice::<Float64Type>(&inputs[1..])?;
    let nrows = data.len() / ncols;

    if (nrows < ncols) || (r <= 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec(name.clone(), vec![f64::NAN]));
    }
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    let ncols_minus_1 = ncols.abs_diff(1);

    // let data_1_view = data.slice(s![..nrows, ..ncols_minus_1]);

    let mut leaves = data
        .chunks_exact(ncols)
        .map(|sl| ((), &sl[..ncols_minus_1]).into())
        .collect::<Vec<_>>();

    // let mut leaves = matrix_to_empty_leaves(&data_1_view);
    let tree = KDT::from_leaves_unchecked(&mut leaves, DIST::LINF);

    let b = if can_parallel {
        data.chunks_exact(ncols)
            .par_bridge()
            .map(|sl| tree.within_count(&sl[..ncols_minus_1], r).unwrap_or(0))
            .sum::<u32>() as f64
            - nrows as f64
    } else {
        data.chunks_exact(ncols).fold(0u32, |acc, sl| {
            acc + tree.within_count(&sl[..ncols_minus_1], r).unwrap_or(0)
        }) as f64
            - nrows as f64
    };

    drop(tree);

    let nrows_minus_1 = nrows.abs_diff(1);
    let mut leaves2 = data
        .chunks_exact(ncols)
        .take(nrows_minus_1)
        .map(|sl| ((), sl).into())
        .collect::<Vec<_>>();

    let tree = KDT::from_leaves_unchecked(&mut leaves2, DIST::LINF);
    let a = if can_parallel {
        data.chunks_exact(ncols)
            .take(nrows_minus_1)
            .par_bridge()
            .map(|sl| tree.within_count(sl, r).unwrap_or(0))
            .sum::<u32>() as f64
            - nrows_minus_1 as f64
    } else {
        data.chunks_exact(ncols)
            .take(nrows_minus_1)
            .fold(0u32, |acc, sl| acc + tree.within_count(sl, r).unwrap_or(0)) as f64
            - nrows_minus_1 as f64
    };
    // Output
    Ok(Series::from_vec(name.clone(), vec![(b / a).ln()]))
}

/// Comptues the logd part of the KNN entropy
fn _knn_entropy_helper<'a>(
    tree: KDT<'a, f64, ()>,
    data: &'a [f64],
    k: usize,
    can_parallel: bool,
) -> f64 {
    let ncols = tree.dim();
    if can_parallel {
        let nrows = data.len() / ncols;
        let splits = split_offsets(nrows, POOL.current_num_threads());
        let partial_sums = splits.into_par_iter().map(|(offset, len)| {
            let subslice = &data[offset * ncols..(offset + len) * ncols];
            subslice.chunks_exact(ncols).fold(0f64, |acc, row| {
                if let Some(mut v) = tree.knn(k + 1, row, 0.) {
                    v.pop()
                        .map(|nb| acc + (2.0 * nb.to_dist()).ln())
                        .unwrap_or(f64::NAN)
                } else {
                    acc
                }
            })
        });
        POOL.install(|| partial_sums.sum())
    } else {
        data.chunks_exact(ncols).fold(0f64, |acc, row| {
            if let Some(mut v) = tree.knn(k + 1, row, 0.) {
                v.pop()
                    .map(|nb| acc + (2.0 * nb.to_dist()).ln())
                    .unwrap_or(f64::NAN)
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
    let ncols = inputs.len();
    let data = series_to_row_major_slice::<Float64Type>(inputs)?;
    let nrows = data.len() / ncols;

    if nrows <= k {
        return Ok(Series::from_vec(name.clone(), vec![f64::NAN]));
    }

    let metric_str = kwargs.metric.as_str();
    let n = nrows as f64;
    let d = ncols as f64;

    // G1
    let g1 = crate::stats_utils::gamma::digamma(n) - crate::stats_utils::gamma::digamma(k as f64);

    // Should support l1, l2, inf here.
    let mut leaves = slice_to_empty_leaves(&data, ncols);
    let (cd, log_d) = if metric_str == "l2" {
        let half_d: f64 = d / 2.0;
        let cd = std::f64::consts::PI.powf(half_d) / (2f64.powf(d)) / (1.0 + half_d).gamma();
        let tree = KDT::from_leaves_unchecked(&mut leaves, DIST::L2);
        (cd, _knn_entropy_helper(tree, &data, k, can_parallel))
    } else if metric_str == "inf" {
        let cd = 1.0;
        let tree = KDT::from_leaves_unchecked(&mut leaves, DIST::LINF);
        (cd, _knn_entropy_helper(tree, &data, k, can_parallel))
    } else {
        return Err(PolarsError::ComputeError(
            "KNN Entropy for distance metric is  not implemented.".into(),
        ));
    };

    let ca = Float64Chunked::from_slice(name.clone(), &[g1 + cd.ln() + log_d * d / n]);
    Ok(ca.into_series())
}
