/// Creating matrix profile for a time series.
/// This is significantly faster than STUMPY when m (window size) is small, and
/// when n (total data points) is large. But is significantly slower when m is large.
use crate::utils::split_offsets;
use kdtree::KdTree;
use num::ToPrimitive;
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use serde::Deserialize;

pub fn matrix_profile_output(_: &[Field]) -> PolarsResult<Field> {
    let dist = Field::new("squared_znorm", DataType::Float64);
    let idx = Field::new("index", DataType::UInt32);
    let v = vec![dist, idx];
    Ok(Field::new("profile", DataType::Struct(v)))
}

#[derive(Deserialize, Debug)]
pub(crate) struct MatrixProfileKwargs {
    pub(crate) window_size: usize,
    pub(crate) leaf_size: usize,
    pub(crate) parallel: bool,
}

// Comments on Performance
// Can be made much faster if kd-tree has add_unchecked, or other ways that bypass the point's finite checks.
// Comments on variations
// We can add a approximate option, which then will still have output of length len - m + 1, but we only
// add x% of data points to the kd-tree. We can easily do this when we add points to tree

#[polars_expr(output_type_func=matrix_profile_output)]
fn pl_matrix_profile(
    inputs: &[Series],
    context: CallerContext,
    kwargs: MatrixProfileKwargs,
) -> PolarsResult<Series> {
    // Set up
    let ts = inputs[0].f64()?;
    let m = kwargs.window_size;
    let mf64 = m as f64;
    let leaf_size = kwargs.leaf_size;
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    if ts.null_count() > 0 || ts.len() <= m || m < 4 {
        return Err(PolarsError::ComputeError(
            "Matrix Profile: Input must not contain nulls and must have length > window size > 4."
                .into(),
        ));
    }
    let ts = ts.rechunk();
    let ts = ts.cont_slice().unwrap();
    let output_size = ts.len() - m + 1;

    // Build the kd-tree
    let mut points: Vec<f64> = Vec::with_capacity(output_size * m);
    // // Creating Z-normalized Points
    // // Init for the rolling stats
    let mut rolling_mean: f64 = ts[..m].iter().sum::<f64>() / mf64;
    let mut old: f64 = ts[0];
    let mut rolling_var = ts[..m]
        .iter()
        .fold(0., |acc, x| acc + (x - rolling_mean).powi(2))
        / (mf64 - 1.0);
    let std = rolling_var.sqrt();
    points.extend(ts[..m].iter().map(|x| (x - rolling_mean) / std));
    // // Build the rolling stats
    for sl in ts[1..].windows(m) {
        let new = sl[m - 1];
        let old_mean = rolling_mean.clone();
        rolling_mean += (new - old) / mf64;
        rolling_var += (new - old) * (new - rolling_mean + old - old_mean) / (mf64 - 1.0);
        old = sl[0];
        let std = rolling_var.sqrt();
        points.extend(sl.iter().map(|x| (x - rolling_mean) / std))
    }
    // // Inserting points into tree
    let mut tree: KdTree<f64, usize, &[f64]> = KdTree::with_capacity(m, leaf_size);
    for (i, p) in points.chunks_exact(m).enumerate() {
        tree.add(p, i)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    }

    // Query and build output
    // Exclusion zone default is: i +- ceil(m/4)
    let exclusion = (mf64 / 4.0).ceil().to_usize().unwrap();

    // Need at least exclusion * 2 + 2 points to ensure at least 1 pt to be outside exclusion. Pigeon hole.
    // We can be faster if we can make k smaller.
    let k = exclusion * 2 + 2;

    let (ca_dist, ca_id) = if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(output_size, n_threads);
            let chunks: (Vec<_>, Vec<_>) = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let start_idx = offset * m;
                    let end_idx = (offset + len) * m;
                    let piece = &points[start_idx..end_idx];
                    let mut dist_builder: PrimitiveChunkedBuilder<Float64Type> =
                        PrimitiveChunkedBuilder::new("", len);
                    let mut idx_builder: PrimitiveChunkedBuilder<UInt32Type> =
                        PrimitiveChunkedBuilder::new("", len);
                    for (i, pt) in piece.chunks_exact(m).enumerate() {
                        let mut dist: Option<f64> = None;
                        let mut id: Option<u32> = None;
                        if let Ok(v) = tree.nearest(&pt, k, &super::squared_euclidean) {
                            for (d, j) in v {
                                if (i + offset).abs_diff(*j) > exclusion {
                                    dist = Some(d);
                                    id = Some(*j as u32);
                                    break;
                                } // <= means in exclusion zone. > means good
                            }
                        }
                        dist_builder.append_option(dist);
                        idx_builder.append_option(id);
                    }
                    let ca1 = dist_builder.finish();
                    let ca2 = idx_builder.finish();
                    (
                        ca1.downcast_iter().cloned().collect::<Vec<_>>(),
                        ca2.downcast_iter().cloned().collect::<Vec<_>>(),
                    )
                })
                .collect();

            (
                Float64Chunked::from_chunk_iter("squared_znorm", chunks.0.into_iter().flatten()),
                UInt32Chunked::from_chunk_iter("index", chunks.1.into_iter().flatten()),
            )
        })
    } else {
        let mut dist_builder: PrimitiveChunkedBuilder<Float64Type> =
            PrimitiveChunkedBuilder::new("squared_znorm", output_size);
        let mut idx_builder: PrimitiveChunkedBuilder<UInt32Type> =
            PrimitiveChunkedBuilder::new("index", output_size);
        for (i, pt) in points.chunks_exact(m).enumerate() {
            let mut dist: Option<f64> = None;
            let mut id: Option<u32> = None;
            if let Ok(v) = tree.nearest(&pt, k, &super::squared_euclidean) {
                for (d, j) in v {
                    if i.abs_diff(*j) > exclusion {
                        dist = Some(d);
                        id = Some(*j as u32);
                        break;
                    } // <= means in exclusion zone. > means good
                }
            }
            dist_builder.append_option(dist);
            idx_builder.append_option(id);
        }
        (dist_builder.finish(), idx_builder.finish())
    };

    let s_dist = ca_dist.into_series();
    // let s_dist = s_dist.extend_constant(AnyValue::Null, m - 1)?;
    let s_id = ca_id.into_series();
    // let s_id = s_id.extend_constant(AnyValue::Null, m - 1)?;
    let out = StructChunked::new("profile", &[s_dist, s_id])?;
    Ok(out.into_series())
}
