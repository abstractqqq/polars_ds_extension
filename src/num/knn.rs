/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
use super::which_distance;
use crate::utils::{list_u64_output, rechunk_to_frame, split_offsets};
use itertools::Itertools;
use kdtree::KdTree;
use ndarray::{s, ArrayView2, Axis};
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct KdtreeKwargs {
    pub(crate) k: usize,
    pub(crate) leaf_size: usize,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
}

#[derive(Deserialize, Debug)]
pub(crate) struct KdtreeRadiusKwargs {
    pub(crate) r: f64,
    pub(crate) leaf_size: usize,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
}

#[inline]
pub fn build_standard_kdtree<'a>(
    dim: usize,
    leaf_size: usize,
    data: &'a ArrayView2<f64>,
) -> Result<KdTree<f64, usize, &'a [f64]>, PolarsError> {
    // Building the tree
    let mut tree = KdTree::with_capacity(dim, leaf_size);
    for (i, p) in data.axis_iter(Axis(0)).enumerate() {
        let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
        let _is_ok = tree
            .add(s, i)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    }
    Ok(tree)
}

pub fn knn_full_output(_: &[Field]) -> PolarsResult<Field> {
    let idx = Field::new("idx", DataType::List(Box::new(DataType::UInt64)));

    let dist = Field::new("dist", DataType::List(Box::new(DataType::Float64)));
    let v = vec![idx, dist];
    Ok(Field::new("knn_w_dist", DataType::Struct(v)))
}

#[polars_expr(output_type_func=list_u64_output)]
fn pl_knn_ptwise(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let id = inputs[0].u64()?;
    let id = id.rechunk();
    let id = id.cont_slice()?;

    let dim = inputs[1..].len();
    if dim == 0 {
        return Err(PolarsError::ComputeError("KNN: No column found.".into()));
    }

    let data = rechunk_to_frame(&inputs[1..])?;
    let nrows = data.height();
    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();

    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Need to use C order because C order is row-contiguous
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding)?;

    // Building output
    let ca = if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let mut builder = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
                        "",
                        len,
                        k + 1,
                        DataType::UInt64,
                    );
                    let piece = data.slice(s![offset..offset + len, 0..dim]);
                    for p in piece.axis_iter(Axis(0)) {
                        let sl = p.to_slice().unwrap();
                        if let Ok(v) = tree.nearest(sl, k + 1, &dist_func) {
                            let s = v.into_iter().map(|(_, i)| id[*i]).collect_vec();
                            builder.append_slice(&s);
                        } else {
                            builder.append_null();
                        }
                    }
                    let ca = builder.finish();
                    ca.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            ListChunked::from_chunk_iter("knn", chunks.into_iter().flatten())
        })
    } else {
        let mut builder =
            ListPrimitiveChunkedBuilder::<UInt64Type>::new("", id.len(), k + 1, DataType::UInt64);

        for p in data.rows() {
            let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(v) = tree.nearest(s, k + 1, &dist_func) {
                let sl = v.into_iter().map(|(_, i)| id[*i]).collect_vec();
                builder.append_slice(&sl);
            } else {
                builder.append_null();
            }
        }
        builder.finish()
    };
    // let ca = builder.finish();
    Ok(ca.into_series())
}

#[polars_expr(output_type_func=list_u64_output)]
fn pl_query_radius_ptwise(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeRadiusKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let id = inputs[0].u64()?;
    let id = id.rechunk();
    let id = id.cont_slice()?;

    let dim = inputs[1..].len();
    if dim == 0 {
        return Err(PolarsError::ComputeError(
            "KNN: No column to decide distance from.".into(),
        ));
    }

    let data = rechunk_to_frame(&inputs[1..])?;
    let nrows = data.height();
    let leaf_size = kwargs.leaf_size;
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    let radius = kwargs.r;
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Need to use C order because C order is row-contiguous
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding)?;

    // Building output
    if can_parallel {
        let ca = POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let mut builder = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
                        "",
                        len,
                        8,
                        DataType::UInt64,
                    );
                    let piece = data.slice(s![offset..offset + len, 0..dim]);
                    for p in piece.axis_iter(Axis(0)) {
                        let sl = p.to_slice().unwrap();
                        if let Ok(v) = tree.within(sl, radius, &dist_func) {
                            let mut out = v.into_iter().map(|(_, i)| id[*i]).collect_vec();
                            out.shrink_to_fit();
                            builder.append_slice(&out);
                        } else {
                            builder.append_null();
                        }
                    }
                    let ca = builder.finish();
                    ca.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            ListChunked::from_chunk_iter("knn-radius", chunks.into_iter().flatten())
        });
        Ok(ca.into_series())
    } else {
        let mut builder =
            ListPrimitiveChunkedBuilder::<UInt64Type>::new("", id.len(), 16, DataType::UInt64);
        for p in data.rows() {
            let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(v) = tree.within(s, radius, &dist_func) {
                let mut out: Vec<u64> = v.into_iter().map(|(_, i)| id[*i]).collect();
                out.shrink_to_fit();
                builder.append_slice(&out);
            } else {
                builder.append_null();
            }
        }
        let ca = builder.finish();
        Ok(ca.into_series())
    }
}

#[polars_expr(output_type_func=knn_full_output)]
fn pl_knn_ptwise_w_dist(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let id = inputs[0].u64()?;
    let id = id.rechunk();
    let id = id.cont_slice().unwrap();

    let dim = inputs[1..].len();
    if dim == 0 {
        return Err(PolarsError::ComputeError(
            "KNN: No column to decide distance from.".into(),
        ));
    }

    let data = rechunk_to_frame(&inputs[1..])?;
    let nrows = data.height();
    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Need to use C order because C order is row-contiguous
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding)?;

    //Building output
    if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: (Vec<_>, Vec<_>) = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let mut nn_builder = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
                        "",
                        len,
                        8,
                        DataType::UInt64,
                    );
                    let mut rr_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                        "",
                        len,
                        8,
                        DataType::Float64,
                    );
                    let piece = data.slice(s![offset..offset + len, 0..dim]);
                    for p in piece.axis_iter(Axis(0)) {
                        let sl = p.to_slice().unwrap();
                        if let Ok(v) = tree.nearest(sl, k + 1, &dist_func) {
                            let mut nn: Vec<u64> = Vec::with_capacity(k + 1);
                            let mut rr: Vec<f64> = Vec::with_capacity(k + 1);
                            //.map(|(_, i)| id[*i]).collect_vec();
                            for (r, i) in v.into_iter() {
                                nn.push(id[*i]);
                                rr.push(r);
                            }
                            nn_builder.append_slice(&nn);
                            rr_builder.append_slice(&rr);
                        } else {
                            nn_builder.append_null();
                            rr_builder.append_null();
                        }
                    }
                    let ca_nn = nn_builder.finish();
                    let ca_rr = rr_builder.finish();
                    (
                        ca_nn.downcast_iter().cloned().collect::<Vec<_>>(),
                        ca_rr.downcast_iter().cloned().collect::<Vec<_>>(),
                    )
                })
                .collect();

            let ca_nn = ListChunked::from_chunk_iter("knn", chunks.0.into_iter().flatten());
            let ca_nn = ca_nn.with_name("knn").into_series();
            let ca_rr = ListChunked::from_chunk_iter("knn-radius", chunks.1.into_iter().flatten());
            let ca_rr = ca_rr.with_name("radius").into_series();
            let out = StructChunked::new("knn_full_output", &[ca_nn, ca_rr])?;
            Ok(out.into_series())
        })
    } else {
        let mut nn_builder =
            ListPrimitiveChunkedBuilder::<UInt64Type>::new("", id.len(), k + 1, DataType::UInt64);

        let mut rr_builder =
            ListPrimitiveChunkedBuilder::<Float64Type>::new("", id.len(), k + 1, DataType::Float64);
        for p in data.rows() {
            let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(v) = tree.nearest(s, k + 1, &dist_func) {
                // By construction, this unwrap is safe
                let mut w_idx: Vec<u64> = Vec::with_capacity(k + 1);
                let mut w_dist: Vec<f64> = Vec::with_capacity(k + 1);
                for (d, i) in v.into_iter() {
                    w_idx.push(id[*i]);
                    w_dist.push(d);
                }
                nn_builder.append_slice(&w_idx);
                rr_builder.append_slice(&w_dist);
            } else {
                nn_builder.append_null();
                rr_builder.append_null();
            }
        }
        let ca_nn = nn_builder.finish();
        let ca_nn = ca_nn.with_name("knn").into_series();
        let ca_rr = rr_builder.finish();
        let ca_rr = ca_rr.with_name("radius").into_series();
        let out = StructChunked::new("knn_full_output", &[ca_nn, ca_rr])?;
        Ok(out.into_series())
    }
}

/// Find all the rows that are the k-nearest neighbors to the point given.
/// Note, only k points will be returned as true, because here the point is considered an "outside" point,
/// not a point in the data.
#[polars_expr(output_type=Boolean)]
fn pl_knn_pt(inputs: &[Series], kwargs: KdtreeKwargs) -> PolarsResult<Series> {
    // Check len
    let pt = inputs[0].f64()?;
    let dim = inputs[1..].len();
    if dim == 0 || pt.len() != dim {
        return Err(PolarsError::ComputeError(
            "KNN: There has to be at least one column in `others` and input point \
            must be the same dimension as the number of columns in `others`."
                .into(),
        ));
    }
    // Set up the point to query
    let binding = pt.rechunk();
    let p = binding.cont_slice()?;
    // Set up params
    let data = rechunk_to_frame(&inputs[1..])?;
    let nrows = data.height();
    let dim = inputs[1..].len();
    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Need to use C order because C order is row-contiguous
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding)?;

    // Building the output
    let mut out: Vec<bool> = vec![false; nrows];
    match tree.nearest(p, k, &dist_func) {
        Ok(v) => {
            for (_, i) in v.into_iter() {
                out[*i] = true;
            }
        }
        Err(e) => {
            return Err(PolarsError::ComputeError(
                ("KNN: ".to_owned() + e.to_string().as_str()).into(),
            ));
        }
    }
    Ok(BooleanChunked::from_slice("", &out).into_series())
}

/// Neighbor count query
#[inline]
pub fn query_nb_cnt<F>(
    tree: &KdTree<f64, usize, &[f64]>,
    data: ArrayView2<f64>,
    dist_func: &F,
    r: f64,
    can_parallel: bool,
) -> UInt32Chunked
where
    F: Fn(&[f64], &[f64]) -> f64 + std::marker::Sync,
{
    if can_parallel {
        let nrows = data.shape()[0];
        let dim = data.shape()[1];
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let piece = data.slice(s![offset..offset + len, 0..dim]);
                    let out = piece.axis_iter(Axis(0)).map(|p| {
                        let sl = p.to_slice().unwrap();
                        if let Ok(cnt) = tree.within_count(sl, r, &dist_func) {
                            Some(cnt as u32)
                        } else {
                            None
                        }
                    });
                    let ca = UInt32Chunked::from_iter_options("", out);
                    ca.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            UInt32Chunked::from_chunk_iter("cnt", chunks.into_iter().flatten())
        })
    } else {
        let output = data.axis_iter(Axis(0)).map(|pt| {
            let s = pt.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(cnt) = tree.within_count(s, r, dist_func) {
                Some(cnt as u32)
            } else {
                None
            }
        });
        UInt32Chunked::from_iter(output)
    }
}

/// For every point in this dataframe, find the number of neighbors within radius r
/// The point itself is always considered as a neighbor to itself.
#[polars_expr(output_type=UInt32)]
fn pl_nb_cnt(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeKwargs,
) -> PolarsResult<Series> {
    // Set up radius
    let radius = inputs[0].f64()?;

    // Set up params
    let dim = inputs[1..].len();
    if dim == 0 {
        return Err(PolarsError::ComputeError(
            "KNN: No column to decide distance from.".into(),
        ));
    }

    let data = rechunk_to_frame(&inputs[1..])?;
    let nrows = data.height();
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    let leaf_size = kwargs.leaf_size;
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;
    // Need to use C order because C order is row-contiguous
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding)?;

    if radius.len() == 1 {
        let r = radius.get(0).unwrap();
        let ca = query_nb_cnt(&tree, data.view(), &dist_func, r, can_parallel);
        Ok(ca.into_series())
    } else if radius.len() == nrows {
        if can_parallel {
            let nrows = data.shape()[0];
            let dim = data.shape()[1];
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(nrows, n_threads);
                let chunks: Vec<_> = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let piece = data.slice(s![offset..offset + len, 0..dim]);
                        let rad = radius.slice(offset as i64, len);
                        let out = piece
                            .axis_iter(Axis(0))
                            .zip(rad.into_iter())
                            .map(|(p, op_r)| {
                                let r = op_r?;
                                let sl = p.to_slice().unwrap();
                                if let Ok(cnt) = tree.within_count(sl, r, &dist_func) {
                                    Some(cnt as u32)
                                } else {
                                    None
                                }
                            });
                        let ca = UInt32Chunked::from_iter_options("", out);
                        ca.downcast_iter().cloned().collect::<Vec<_>>()
                    })
                    .collect();

                let ca = UInt32Chunked::from_chunk_iter("cnt", chunks.into_iter().flatten());
                Ok(ca.into_series())
            })
        } else {
            let ca = UInt32Chunked::from_iter(radius.into_iter().zip(data.axis_iter(Axis(0))).map(
                |(rad, pt)| {
                    let r = rad?;
                    let s = pt.to_slice().unwrap(); // C order makes sure rows are contiguous
                    if let Ok(cnt) = tree.within_count(s, r, &dist_func) {
                        Some(cnt as u32)
                    } else {
                        None
                    }
                },
            ));
            Ok(ca.into_series())
        }
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
