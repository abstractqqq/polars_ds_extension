/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
use super::which_distance;
use crate::{
    arkadia::{
        arkadia_lp::LpKdtree, matrix_to_leaves, matrix_to_leaves_w_norm,
        utils::matrix_to_empty_leaves_w_norm, SplitMethod, KDTQ,
    },
    utils::{list_u32_output, rechunk_to_frame, series_to_ndarray, split_offsets},
};
use itertools::Itertools;
use kdtree::KdTree;
use ndarray::{s, ArrayView2, Axis};
use num::Float;
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    error::ComputeError,
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use serde::Deserialize;

pub fn knn_full_output(_: &[Field]) -> PolarsResult<Field> {
    let idx = Field::new("idx", DataType::List(Box::new(DataType::UInt32)));
    let dist = Field::new("dist", DataType::List(Box::new(DataType::Float64)));
    let v = vec![idx, dist];
    Ok(Field::new("knn_dist", DataType::Struct(v)))
}

#[derive(Deserialize, Debug)]
pub(crate) struct KdtreeKwargs {
    pub(crate) k: usize,
    pub(crate) leaf_size: usize,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
    pub(crate) skip_eval: bool,
    pub(crate) skip_data: bool,
}

#[derive(Deserialize, Debug)]
pub(crate) struct KdtreeRadiusKwargs {
    pub(crate) r: f64,
    pub(crate) leaf_size: usize,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
}

pub fn build_knn_matrix_data(features: &[Series]) -> PolarsResult<ndarray::Array2<f64>> {
    let data = rechunk_to_frame(features)?;
    data.to_ndarray::<Float64Type>(IndexOrder::C)
}

/// Builds a standard kd-tree
/// If keep is None, add all points.
/// If keep is some bool vec, keep only the points where it is true
#[inline]
pub fn build_standard_kdtree<'a>(
    dim: usize,
    leaf_size: usize,
    data: &'a ArrayView2<f64>,
    keep: Option<Vec<bool>>,
) -> PolarsResult<KdTree<f64, usize, &'a [f64]>> {
    if dim == 0 {
        return Err(PolarsError::ComputeError("KNN: No column found.".into()));
    }

    // Building the tree
    // C order makes sure rows are contiguous.
    // Add can have error. If error, then ignore the addition of that row.
    let mut tree = KdTree::with_capacity(dim, leaf_size);
    if let Some(b) = keep {
        for (i, p) in data.axis_iter(Axis(0)).enumerate().filter(|(i, _)| b[*i]) {
            let _ = tree.add(p.to_slice().unwrap(), i);
        }
    } else {
        for (i, p) in data.axis_iter(Axis(0)).enumerate() {
            let _ = tree.add(p.to_slice().unwrap(), i);
        }
    };
    Ok(tree)
}

#[polars_expr(output_type_func=list_u32_output)]
fn pl_knn_ptwise(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    let can_parallel = kwargs.parallel && !context.parallel();
    let skip_eval = kwargs.skip_eval;
    let skip_data = kwargs.skip_data;

    let mut inputs_offset = 0;
    let id = inputs[inputs_offset].u32().unwrap();
    let id = id.cont_slice()?;
    // eval_mask
    // Skip evaluation for certain rows. This is helpful when K is large and only KNN for certain
    // rows is required.
    let eval_mask = if skip_eval {
        inputs_offset += 1; // True means evaluate.
        let eval_mask = inputs[inputs_offset].bool()?;
        Some(
            eval_mask
                .iter()
                .map(|b| b.unwrap_or(false))
                .collect::<Vec<bool>>(),
        )
    } else {
        None
    };
    let eval_mask = eval_mask.as_ref();

    // data_mask
    // Use only the true rows as neighbors. None true rows cannot be nearest neighbors, despite close distance.
    let data_mask = if skip_data {
        inputs_offset += 1;
        let data_mask = inputs[inputs_offset].bool()?;
        Some(
            data_mask
                .iter()
                .map(|b| b.unwrap_or(false))
                .collect::<Vec<bool>>(),
        )
    } else {
        None
    };

    let data = build_knn_matrix_data(&inputs[inputs_offset + 1..])?;
    let nrows = data.nrows();
    let dim = data.ncols();
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding, data_mask)?;

    // Building output
    let ca = if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder =
                ListPrimitiveChunkedBuilder::<UInt32Type>::new("", len, k + 1, DataType::UInt32);
            let piece = data.slice(s![offset..offset + len, 0..dim]);
            for (i, p) in piece.axis_iter(Axis(0)).enumerate() {
                if eval_mask.map(|f| f[i + offset]).unwrap_or(true) {
                    let s = p.to_slice().unwrap();
                    match tree.nearest(s, k + 1, &dist_func) {
                        Ok(v) => {
                            let s = v.into_iter().map(|(_, i)| id[*i]).collect_vec();
                            builder.append_slice(&s)
                        }
                        Err(_) => builder.append_null(),
                    }
                } else {
                    builder.append_null();
                };
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });

        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        ListChunked::from_chunk_iter("knn", chunks.into_iter().flatten())
    } else {
        let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
            "knn",
            id.len(),
            k + 1,
            DataType::UInt32,
        );
        for (i, p) in data.rows().into_iter().enumerate() {
            if eval_mask.map(|f| f[i]).unwrap_or(true) {
                // evaluate this point
                let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
                match tree.nearest(s, k + 1, &dist_func) {
                    Ok(v) => {
                        let sl = v.into_iter().map(|(_, i)| id[*i]).collect_vec();
                        builder.append_slice(&sl);
                    }
                    Err(_) => builder.append_null(),
                }
            } else {
                // ignore this point
                builder.append_null()
            }
        }
        builder.finish()
    };
    Ok(ca.into_series())
}

#[polars_expr(output_type_func=list_u32_output)]
fn pl_query_radius_ptwise(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KdtreeRadiusKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let leaf_size = kwargs.leaf_size;
    let can_parallel = kwargs.parallel && !context.parallel();
    let radius = kwargs.r;

    let id = inputs[0].u32()?;
    let id = id.cont_slice()?;

    let data = build_knn_matrix_data(&inputs[1..])?;
    let nrows = data.nrows();
    let dim = data.ncols();
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Building the tree
    // let binding = data.view();
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding, None)?;

    // Building output
    if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder =
                ListPrimitiveChunkedBuilder::<UInt32Type>::new("", len, 8, DataType::UInt32);
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
        });

        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        let ca = ListChunked::from_chunk_iter("knn-radius", chunks.into_iter().flatten());
        Ok(ca.into_series())
    } else {
        let mut builder =
            ListPrimitiveChunkedBuilder::<UInt32Type>::new("", id.len(), 16, DataType::UInt32);
        for p in data.rows() {
            let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(v) = tree.within(s, radius, &dist_func) {
                let mut out: Vec<u32> = v.into_iter().map(|(_, i)| id[*i]).collect();
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
    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    let can_parallel = kwargs.parallel && !context.parallel();
    let skip_eval = kwargs.skip_eval;
    let skip_data = kwargs.skip_data;

    let mut inputs_offset = 0;
    let id = inputs[inputs_offset].u32()?;
    let id = id.cont_slice()?;
    // eval_mask
    // Skip evaluation for certain rows. This is helpful when K is large and only KNN for certain
    // rows is required.
    let eval_mask = if skip_eval {
        inputs_offset += 1; // True means evaluate.
        let eval_mask = inputs[inputs_offset].bool()?;
        Some(
            eval_mask
                .iter()
                .map(|b| b.unwrap_or(false))
                .collect::<Vec<bool>>(),
        )
    } else {
        None
    };
    let eval_mask = eval_mask.as_ref();

    // data_mask
    // Use only the true rows as neighbors. None true rows cannot be nearest neighbors, despite close distance.
    let data_mask = if skip_data {
        inputs_offset += 1;
        let data_mask = inputs[inputs_offset].bool()?;
        Some(
            data_mask
                .iter()
                .map(|b| b.unwrap_or(false))
                .collect::<Vec<bool>>(),
        )
    } else {
        None
    };

    let data = build_knn_matrix_data(&inputs[inputs_offset + 1..])?;
    let nrows = data.nrows();
    let dim = data.ncols();
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding, data_mask)?;

    //Building output
    if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: (Vec<_>, Vec<_>) = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let mut nn_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
                        "",
                        len,
                        k + 1,
                        DataType::UInt32,
                    );
                    let mut rr_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                        "",
                        len,
                        k + 1,
                        DataType::Float64,
                    );
                    let piece = data.slice(s![offset..offset + len, 0..dim]);
                    for (i, p) in piece.axis_iter(Axis(0)).enumerate() {
                        if eval_mask.map(|f| f[i + offset]).unwrap_or(true) {
                            let s = p.to_slice().unwrap();
                            match tree.nearest(s, k + 1, &dist_func) {
                                Ok(v) => {
                                    let mut nn: Vec<u32> = Vec::with_capacity(k + 1);
                                    let mut rr: Vec<f64> = Vec::with_capacity(k + 1);
                                    for (r, i) in v.into_iter() {
                                        nn.push(id[*i]);
                                        rr.push(r);
                                    }
                                    nn_builder.append_slice(&nn);
                                    rr_builder.append_slice(&rr);
                                }
                                Err(_) => {
                                    nn_builder.append_null();
                                    rr_builder.append_null();
                                }
                            }
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

            let ca_nn = ListChunked::from_chunk_iter("idx", chunks.0.into_iter().flatten());
            let ca_nn = ca_nn.into_series();
            let ca_rr = ListChunked::from_chunk_iter("dist", chunks.1.into_iter().flatten());
            let ca_rr = ca_rr.into_series();
            let out = StructChunked::new("knn_dist", &[ca_nn, ca_rr])?;
            Ok(out.into_series())
        })
    } else {
        let mut nn_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
            "idx",
            id.len(),
            k + 1,
            DataType::UInt32,
        );

        let mut rr_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            "dist",
            id.len(),
            k + 1,
            DataType::Float64,
        );
        for (i, p) in data.rows().into_iter().enumerate() {
            if eval_mask.map(|f| f[i]).unwrap_or(true) {
                let s = p.to_slice().unwrap();
                match tree.nearest(s, k + 1, &dist_func) {
                    Ok(v) => {
                        let mut nn: Vec<u32> = Vec::with_capacity(k + 1);
                        let mut rr: Vec<f64> = Vec::with_capacity(k + 1);
                        for (r, i) in v.into_iter() {
                            nn.push(id[*i]);
                            rr.push(r);
                        }
                        nn_builder.append_slice(&nn);
                        rr_builder.append_slice(&rr);
                    }
                    Err(_) => {
                        nn_builder.append_null();
                        rr_builder.append_null();
                    }
                }
            } else {
                nn_builder.append_null();
                rr_builder.append_null();
            }
        }
        let ca_nn = nn_builder.finish();
        let ca_nn = ca_nn.into_series();
        let ca_rr = rr_builder.finish();
        let ca_rr = ca_rr.into_series();
        let out = StructChunked::new("knn_dist", &[ca_nn, ca_rr])?;
        Ok(out.into_series())
    }
}

/// Find all rows that are the k-nearest neighbors to the point given.
/// Note, only k points will be returned as true, because here the point is considered an "outside" point,
/// not a point in the data.
#[polars_expr(output_type=Boolean)]
fn pl_knn_filter(inputs: &[Series], kwargs: KdtreeKwargs) -> PolarsResult<Series> {
    // Set up params
    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    // Check len
    let data = build_knn_matrix_data(&inputs[1..])?;
    let nrows = data.nrows();
    let pt = inputs[0].f64()?;
    let p = pt.cont_slice()?;
    let dim = inputs[1..].len();
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;
    if pt.len() != dim {
        return Err(PolarsError::ComputeError(
            "KNN: input point must have the same dim as the number of columns in `others`.".into(),
        ));
    }
    // Set up the point to query

    // Building the tree
    // let binding = data.view();
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding, None)?;

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


#[inline]
pub fn query_nb_cnt<'a, Kdt>(
    tree: Kdt,
    data: ArrayView2<'a, f64>,
    r: f64,
    can_parallel: bool,
) -> UInt32Chunked
where
    Kdt: KDTQ<'a, f64, ()> + std::marker::Sync,
{
    // as_slice.unwrap() is safe because when we create the matrices, we specified C order.
    let nrows = data.nrows();
    let ncols = data.ncols();
    if can_parallel {
        let splits = split_offsets(nrows, POOL.current_num_threads());
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder: PrimitiveChunkedBuilder<UInt32Type> =
                PrimitiveChunkedBuilder::new("", nrows);

            let piece = data.slice(s![offset..offset + len, ..ncols]);
            for row in piece.rows() {
                builder.append_option(tree.within_count(row.to_slice().unwrap(), r));
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        UInt32Chunked::from_chunk_iter("cnt", chunks.into_iter().flatten())
    } else {
        UInt32Chunked::from_iter_options(
            "cnt",
            data.rows()
                .into_iter()
                .map(|row| tree.within_count(row.to_slice().unwrap(), r)),
        )
    }
}

#[inline]
pub fn query_nb_cnt_many_radius<'a, Kdt>(
    tree: Kdt,
    data: ArrayView2<'a, f64>,
    radius: Float64Chunked,
    can_parallel: bool,
) -> UInt32Chunked
where
    Kdt: KDTQ<'a, f64, ()> + std::marker::Sync,
{
    // as_slice.unwrap() is safe because when we create the matrices, we specified C order.
    if can_parallel {
        let radius = radius.to_vec();
        let nrows = data.nrows();
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder: PrimitiveChunkedBuilder<UInt32Type> =
                PrimitiveChunkedBuilder::new("", nrows);
            let piece = data.slice(s![offset..offset + len, ..]);
            let rad = &radius[offset..offset + len];
            for (row, r) in piece.rows().into_iter().zip(rad.iter()) {
                match r {
                    Some(r) => builder.append_option(tree.within_count(row.as_slice().unwrap(), *r)),
                    None => builder.append_null(),
                }
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        UInt32Chunked::from_chunk_iter("cnt", chunks.into_iter().flatten())
    } else {
        UInt32Chunked::from_iter_options(
            "cnt", 
            data.rows().into_iter()
            .zip(radius.into_iter())
            .map(|(row, r)| {
                match r {
                    Some(r) => tree.within_count(row.as_slice().unwrap(), r),
                    None => None,
                }
            })
        )
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
    // Set up params
    let can_parallel = kwargs.parallel && !context.parallel();
    // let leaf_size = kwargs.leaf_size;
    // Set up radius
    let radius = inputs[0].f64()?;

    let data = series_to_ndarray(&inputs[1..], IndexOrder::C)?;
    let nrows = data.nrows();

    let binding = data.view();
    let mut leaves = matrix_to_empty_leaves_w_norm(&binding);

    // always l2 for now
    let tree = crate::arkadia::arkadia::Kdtree::from_leaves(&mut leaves, SplitMethod::MIDPOINT)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    if radius.len() == 1 {
        let r = radius.get(0).unwrap();
        let ca = query_nb_cnt(tree, data.view(), r, can_parallel);
        Ok(ca.with_name("cnt").into_series())
    } else if radius.len() == nrows {
        let ca = if can_parallel {
            let radius = radius.to_vec();
            let nrows = data.nrows();
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
                let mut builder: PrimitiveChunkedBuilder<UInt32Type> =
                    PrimitiveChunkedBuilder::new("", nrows);
                let piece = data.slice(s![offset..offset + len, ..]);
                let rad = &radius[offset..offset + len];
                for (row, r) in piece.rows().into_iter().zip(rad.iter()) {
                    match r {
                        Some(r) => builder.append_option(tree.within_count(row.as_slice().unwrap(), *r)),
                        None => builder.append_null(),
                    }
                }
                let ca = builder.finish();
                ca.downcast_iter().cloned().collect::<Vec<_>>()
            });
            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            UInt32Chunked::from_chunk_iter("cnt", chunks.into_iter().flatten())
        } else {
            UInt32Chunked::from_iter_options(
                "cnt", 
                data.rows().into_iter()
                .zip(radius.into_iter())
                .map(|(row, r)| {
                    match r {
                        Some(r) => tree.within_count(row.as_slice().unwrap(), r),
                        None => None,
                    }
                })
            )
        };
        Ok(ca.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
