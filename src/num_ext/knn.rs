/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
use crate::{
    arkadia::{
        matrix_to_empty_leaves, matrix_to_leaves, AnyKDT, KNNMethod, KNNRegressor, Leaf,
        SpacialQueries,
    },
    utils::{list_u32_output, series_to_ndarray, split_offsets, DIST},
};

use ndarray::{s, ArrayView2};
use num::Float;
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
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

#[derive(Deserialize)]
pub(crate) struct KNNAvgKwargs {
    // pub(crate) leaf_size: usize,
    pub(crate) k: usize,
    pub(crate) metric: String,
    #[serde(default)]
    pub(crate) weighted: bool,
    #[serde(default)]
    pub(crate) parallel: bool,
    pub(crate) min_bound: f64,
    #[serde(default = "_max_bound")]
    pub(crate) max_bound: f64,
}

#[derive(Deserialize)]
pub(crate) struct KDTKwargs {
    // pub(crate) leaf_size: usize,
    pub(crate) k: usize,
    pub(crate) metric: String,
    #[serde(default)]
    pub(crate) parallel: bool,
    #[serde(default)]
    pub(crate) skip_eval: bool,
    #[serde(default = "_max_bound")]
    pub(crate) max_bound: f64,
    #[serde(default)]
    pub(crate) epsilon: f64,
}

fn _max_bound() -> f64 {
    f64::max_value()
}

#[derive(Deserialize, Debug)]
pub(crate) struct KDTRadiusKwargs {
    // pub(crate) leaf_size: usize,
    pub(crate) r: f64,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
    pub(crate) sort: bool,
}

pub fn matrix_to_leaves_filtered<'a, T: Float + 'static, A: Copy>(
    matrix: &'a ArrayView2<'a, T>,
    values: &'a [A],
    filter: &BooleanChunked,
) -> Vec<Leaf<'a, T, A>> {
    filter
        .into_iter()
        .zip(values.iter().copied().zip(matrix.rows()))
        .filter(|(f, _)| f.unwrap_or(false))
        .map(|(_, pair)| pair.into())
        .collect::<Vec<_>>()
}

// used in all cases but squared l2 (multiple queries)
pub fn dist_from_str<T: Float + cfavml::safe_trait_distance_ops::DistanceOps + 'static>(
    dist_str: String,
) -> Result<DIST<T>, String> {
    match dist_str.as_ref() {
        "l1" => Ok(DIST::L1),
        "l2" => Ok(DIST::L2),
        "sql2" => Ok(DIST::SQL2),
        "linf" | "inf" => Ok(DIST::LINF),
        "cosine" => Ok(DIST::ANY(cfavml::cosine)),
        _ => Err("Unknown distance metric.".into()),
    }
}

/// KNN Regression
/// Always do k + 1 because this operation is in-dataframe, and this means
/// that the point itself is always a neighbor to itself.
#[polars_expr(output_type=Float64)]
fn pl_knn_avg(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KNNAvgKwargs,
) -> PolarsResult<Series> {
    // Set up params

    // let leaf_size = kwargs.leaf_size;
    let k = kwargs.k;
    let can_parallel = kwargs.parallel && !context.parallel();
    let max_bound = kwargs.max_bound;
    let min_bound = kwargs.min_bound;
    let method = KNNMethod::new(kwargs.weighted, min_bound);

    let id = inputs[0].f64().unwrap();
    let id = id.cont_slice()?;
    let null_mask = inputs[1].bool().unwrap();
    let nrows = null_mask.len();

    let data = series_to_ndarray(&inputs[2..], IndexOrder::C)?;
    let binding = data.view();
    let mut leaves = matrix_to_leaves_filtered(&binding, id, &null_mask);

    let tree = match dist_from_str::<f64>(kwargs.metric) {
        Ok(d) => Ok(AnyKDT::from_leaves_unchecked(&mut leaves, d)),
        Err(e) => Err(e),
    }
    .map_err(|err| PolarsError::ComputeError(err.into()))?;

    let ca = if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let slice = binding.slice(s![offset..offset + len, ..]);
                    let out = Float64Chunked::from_iter_options(
                        "",
                        slice.rows().into_iter().map(|row| {
                            let sl = row.as_slice().unwrap();
                            tree.knn_regress(k + 1, sl, min_bound, max_bound, method)
                        }),
                    );
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();

            Float64Chunked::from_chunk_iter("", chunks.into_iter().flatten())
        })
    } else {
        Float64Chunked::from_iter_options(
            "",
            data.rows().into_iter().map(|row| {
                let sl = row.as_slice().unwrap();
                tree.knn_regress(k + 1, sl, min_bound, max_bound, method)
            }),
        )
    };

    Ok(ca.into_series())
}

/// KNN Point-wise
/// Always do k + 1 because this operation is in-dataframe, and this means
/// that the point itself is always a neighbor to itself.
pub fn knn_ptwise<'a, Kdt>(
    tree: Kdt,
    eval_mask: Vec<bool>,
    data: ArrayView2<'a, f64>,
    k: usize,
    can_parallel: bool,
    max_bound: f64,
    epsilon: f64,
) -> ListChunked
where
    Kdt: SpacialQueries<'a, f64, u32> + std::marker::Sync,
{
    let nrows = data.nrows();
    if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder =
                ListPrimitiveChunkedBuilder::<UInt32Type>::new("", len, k + 1, DataType::UInt32);

            let piece = data.slice(s![offset..offset + len, ..]);
            let mask = &eval_mask[offset..offset + len];
            for (b, p) in mask.iter().zip(piece.rows()) {
                if *b {
                    match tree.knn_bounded(k + 1, p.to_slice().unwrap(), max_bound, epsilon) {
                        Some(nbs) => {
                            let v = nbs.into_iter().map(|nb| nb.to_item()).collect::<Vec<u32>>();
                            builder.append_slice(&v);
                        }
                        None => builder.append_null(),
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
            eval_mask.len(),
            k + 1,
            DataType::UInt32,
        );
        for (b, p) in eval_mask.into_iter().zip(data.rows()) {
            if b {
                match tree.knn_bounded(k + 1, p.to_slice().unwrap(), max_bound, epsilon) {
                    Some(nbs) => {
                        let v = nbs.into_iter().map(|nb| nb.to_item()).collect::<Vec<u32>>();
                        builder.append_slice(&v);
                    }
                    None => builder.append_null(),
                }
            } else {
                builder.append_null();
            }
        }
        builder.finish()
    }
}

#[polars_expr(output_type_func=list_u32_output)]
fn pl_knn_ptwise(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let k = kwargs.k;
    // let leaf_size = kwargs.leaf_size;
    let can_parallel = kwargs.parallel && !context.parallel();
    let skip_eval = kwargs.skip_eval;

    let id = inputs[0].u32().unwrap();
    let id = id.cont_slice()?;
    let nrows = id.len();

    // True means no nulls, keep
    let null_mask = inputs[1].bool().unwrap();

    let mut inputs_offset = 2;
    let eval_mask = if skip_eval {
        let eval_mask = inputs[2].bool().unwrap();
        inputs_offset = 3;
        eval_mask
            .iter()
            .map(|b| b.unwrap_or(false))
            .collect::<Vec<bool>>()
    } else {
        vec![true; nrows]
    };

    let data = series_to_ndarray(&inputs[inputs_offset..], IndexOrder::C)?;
    let binding = data.view();

    let ca = match dist_from_str::<f64>(kwargs.metric) {
        Ok(d) => {
            let mut leaves = matrix_to_leaves_filtered(&binding, id, null_mask);
            let tree = AnyKDT::from_leaves_unchecked(&mut leaves, d);
            Ok(knn_ptwise(
                tree,
                eval_mask,
                binding,
                k,
                can_parallel,
                kwargs.max_bound,
                kwargs.epsilon,
            ))
        }
        Err(e) => Err(e),
    }
    .map_err(|err| PolarsError::ComputeError(err.into()))?;

    Ok(ca.into_series())
}

pub fn knn_ptwise_w_dist<'a, Kdt>(
    tree: Kdt,
    eval_mask: Vec<bool>,
    data: ArrayView2<'a, f64>,
    k: usize,
    can_parallel: bool,
    max_bound: f64,
    epsilon: f64,
) -> (ListChunked, ListChunked)
where
    Kdt: SpacialQueries<'a, f64, u32> + std::marker::Sync,
{
    let nrows = data.nrows();
    if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: (Vec<_>, Vec<_>) = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
                        "",
                        len,
                        k + 1,
                        DataType::UInt32,
                    );
                    let mut dist_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                        "",
                        len,
                        k + 1,
                        DataType::Float64,
                    );
                    let piece = data.slice(s![offset..offset + len, ..]);
                    let mask = &eval_mask[offset..offset + len];
                    for (b, p) in mask.iter().zip(piece.rows()) {
                        if *b {
                            match tree.knn_bounded(k + 1, p.to_slice().unwrap(), max_bound, epsilon)
                            {
                                Some(nbs) => {
                                    let mut distances = Vec::with_capacity(nbs.len());
                                    let mut neighbors = Vec::with_capacity(nbs.len());
                                    for (d, id) in nbs.into_iter().map(|nb| nb.to_pair()) {
                                        distances.push(d);
                                        neighbors.push(id);
                                    }
                                    builder.append_slice(&neighbors);
                                    dist_builder.append_slice(&distances);
                                }
                                None => {
                                    builder.append_null();
                                    dist_builder.append_null();
                                }
                            }
                        } else {
                            builder.append_null();
                            dist_builder.append_null();
                        }
                    }
                    let ca_nb = builder.finish();
                    let ca_dist = dist_builder.finish();
                    (
                        ca_nb.downcast_iter().cloned().collect::<Vec<_>>(),
                        ca_dist.downcast_iter().cloned().collect::<Vec<_>>(),
                    )
                })
                .collect();

            let ca_nb = ListChunked::from_chunk_iter("idx", chunks.0.into_iter().flatten());
            let ca_dist = ListChunked::from_chunk_iter("dist", chunks.1.into_iter().flatten());
            (ca_nb, ca_dist)
        })
    } else {
        let mut builder =
            ListPrimitiveChunkedBuilder::<UInt32Type>::new("idx", nrows, k + 1, DataType::UInt32);
        let mut dist_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            "dist",
            nrows,
            k + 1,
            DataType::Float64,
        );
        for (b, p) in eval_mask.into_iter().zip(data.rows()) {
            if b {
                match tree.knn_bounded(k + 1, p.to_slice().unwrap(), max_bound, epsilon) {
                    Some(nbs) => {
                        // let v = nbs.into_iter().map(|nb| nb.to_item()).collect::<Vec<u32>>();
                        // builder.append_slice(&v);
                        let mut distances = Vec::with_capacity(nbs.len());
                        let mut neighbors = Vec::with_capacity(nbs.len());
                        for (d, id) in nbs.into_iter().map(|nb| nb.to_pair()) {
                            distances.push(d);
                            neighbors.push(id);
                        }
                        builder.append_slice(&neighbors);
                        dist_builder.append_slice(&distances);
                    }
                    None => {
                        builder.append_null();
                        dist_builder.append_null();
                    }
                }
            } else {
                builder.append_null();
                dist_builder.append_null();
            }
        }
        (builder.finish(), dist_builder.finish())
    }
}

#[polars_expr(output_type_func=knn_full_output)]
fn pl_knn_ptwise_w_dist(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let k = kwargs.k;
    // let leaf_size = kwargs.leaf_size;
    let can_parallel = kwargs.parallel && !context.parallel();
    let skip_eval = kwargs.skip_eval;

    let id = inputs[0].u32().unwrap();
    let id = id.cont_slice()?;
    let nrows = id.len();

    let null_mask = inputs[1].bool().unwrap();

    let mut inputs_offset = 2;
    let eval_mask = if skip_eval {
        let eval_mask = inputs[2].bool().unwrap();
        inputs_offset = 3;
        eval_mask
            .iter()
            .map(|b| b.unwrap_or(false))
            .collect::<Vec<bool>>()
    } else {
        vec![true; nrows]
    };

    let data = series_to_ndarray(&inputs[inputs_offset..], IndexOrder::C)?;
    let binding = data.view();

    let (ca_nb, ca_dist) = match dist_from_str::<f64>(kwargs.metric) {
        Ok(d) => {
            let mut leaves = matrix_to_leaves_filtered(&binding, id, null_mask);
            let tree = AnyKDT::from_leaves_unchecked(&mut leaves, d);
            Ok(knn_ptwise_w_dist(
                tree,
                eval_mask,
                binding,
                k,
                can_parallel,
                kwargs.max_bound,
                kwargs.epsilon,
            ))
        }
        Err(e) => Err(e),
    }
    .map_err(|err| PolarsError::ComputeError(err.into()))?;

    let out = StructChunked::new("knn_dist", &[ca_nb.into_series(), ca_dist.into_series()])?;
    Ok(out.into_series())
}

pub fn query_radius_ptwise<'a, Kdt>(
    tree: Kdt,
    data: ArrayView2<'a, f64>,
    r: f64,
    can_parallel: bool,
    sort: bool,
) -> ListChunked
where
    Kdt: SpacialQueries<'a, f64, u32> + std::marker::Sync,
{
    if can_parallel {
        let nrows = data.nrows();
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder =
                ListPrimitiveChunkedBuilder::<UInt32Type>::new("", len, 16, DataType::UInt32);
            let piece = data.slice(s![offset..offset + len, ..]);
            for p in piece.rows() {
                let sl = p.to_slice().unwrap();
                if let Some(v) = tree.within(sl, r, sort) {
                    let out: Vec<u32> = v.into_iter().map(|nb| nb.to_item()).collect();
                    builder.append_slice(&out);
                } else {
                    builder.append_null();
                }
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        ListChunked::from_chunk_iter("", chunks.into_iter().flatten())
    } else {
        let mut builder =
            ListPrimitiveChunkedBuilder::<UInt32Type>::new("", data.len(), 16, DataType::UInt32);

        for p in data.rows() {
            let sl = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Some(v) = tree.within(sl, r, sort) {
                let out: Vec<u32> = v.into_iter().map(|nb| nb.to_item()).collect();
                builder.append_slice(&out);
            } else {
                builder.append_null();
            }
        }
        builder.finish()
    }
}

#[polars_expr(output_type_func=list_u32_output)]
fn pl_query_radius_ptwise(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTRadiusKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let can_parallel = kwargs.parallel && !context.parallel();
    let radius = kwargs.r;
    let sort = kwargs.sort;

    let id = inputs[0].u32()?;
    let id = id.cont_slice()?;

    let data = series_to_ndarray(&inputs[1..], IndexOrder::C)?;
    let binding = data.view();
    // Building output

    let ca = match dist_from_str::<f64>(kwargs.metric) {
        Ok(d) => {
            let mut leaves = matrix_to_leaves(&binding, id);
            let tree = AnyKDT::from_leaves_unchecked(&mut leaves, d);
            Ok(query_radius_ptwise(
                tree,
                binding,
                radius,
                can_parallel,
                sort,
            ))
        }
        Err(e) => Err(e),
    }
    .map_err(|err| PolarsError::ComputeError(err.into()))?;
    Ok(ca.into_series())
}

#[inline]
pub fn query_nb_cnt<'a, Kdt>(
    tree: Kdt,
    data: ArrayView2<'a, f64>,
    r: f64,
    can_parallel: bool,
) -> UInt32Chunked
where
    Kdt: SpacialQueries<'a, f64, ()> + std::marker::Sync,
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
pub fn query_nb_cnt_w_radius<'a, Kdt>(
    tree: Kdt,
    data: ArrayView2<'a, f64>,
    radius: &Float64Chunked,
    can_parallel: bool,
) -> UInt32Chunked
where
    Kdt: SpacialQueries<'a, f64, ()> + std::marker::Sync,
{
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
                    Some(r) => {
                        builder.append_option(tree.within_count(row.as_slice().unwrap(), *r))
                    }
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
            data.rows()
                .into_iter()
                .zip(radius.into_iter())
                .map(|(row, r)| match r {
                    Some(r) => tree.within_count(row.as_slice().unwrap(), r),
                    None => None,
                }),
        )
    }
}

/// For every point in this dataframe, find the number of neighbors within radius r
/// The point itself is always considered as a neighbor to itself.
#[polars_expr(output_type=UInt32)]
fn pl_nb_cnt(inputs: &[Series], context: CallerContext, kwargs: KDTKwargs) -> PolarsResult<Series> {
    let radius = inputs[0].f64()?;
    let can_parallel = kwargs.parallel && !context.parallel();

    let data = series_to_ndarray(&inputs[1..], IndexOrder::C)?;
    let nrows = data.nrows();

    let binding = data.view();
    if radius.len() == 1 {
        let r = radius.get(0).unwrap();
        let ca = match dist_from_str::<f64>(kwargs.metric) {
            Ok(d) => {
                let mut leaves = matrix_to_empty_leaves(&binding);
                let tree = AnyKDT::from_leaves_unchecked(&mut leaves, d);
                Ok(query_nb_cnt(tree, data.view(), r, can_parallel))
            }
            Err(e) => Err(e),
        }
        .map_err(|err| PolarsError::ComputeError(err.into()))?;
        Ok(ca.with_name("cnt").into_series())
    } else if radius.len() == nrows {
        let ca = match dist_from_str::<f64>(kwargs.metric) {
            Ok(d) => {
                let mut leaves = matrix_to_empty_leaves(&binding);
                let tree = AnyKDT::from_leaves_unchecked(&mut leaves, d);
                Ok(query_nb_cnt_w_radius(
                    tree,
                    data.view(),
                    radius,
                    can_parallel,
                ))
            }
            Err(e) => Err(e),
        }
        .map_err(|err| PolarsError::ComputeError(err.into()))?;
        Ok(ca.with_name("cnt").into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
