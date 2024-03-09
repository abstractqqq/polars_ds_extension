use crate::utils::split_offsets;
use itertools::Itertools;
use petgraph::algo::{astar, dijkstra};
use petgraph::visit::IntoNodeReferences;
use petgraph::Directed;
use petgraph::{stable_graph::NodeIndex, Graph};
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};

// What is the proper heuristic to use?

pub fn dijkstra_output(_: &[Field]) -> PolarsResult<Field> {
    let node = Field::new("node", DataType::UInt32);
    let reachable = Field::new("reachable", DataType::Boolean);
    let steps = Field::new("steps", DataType::UInt32);
    let v = vec![node, reachable, steps];
    Ok(Field::new("", DataType::Struct(v)))
}

pub fn shortest_path_const_cost_output(_: &[Field]) -> PolarsResult<Field> {
    let node = Field::new("node", DataType::UInt32);
    let path = Field::new("path", DataType::List(Box::new(DataType::UInt32)));
    let v = vec![node, path];
    Ok(Field::new("shortest_path", DataType::Struct(v)))
}

pub fn shortest_path_output(_: &[Field]) -> PolarsResult<Field> {
    let node = Field::new("node", DataType::UInt32);
    let path = Field::new("path", DataType::List(Box::new(DataType::UInt64)));
    let cost = Field::new("cost", DataType::Float64);
    let v = vec![node, path, cost];
    Ok(Field::new("shortest_path", DataType::Struct(v)))
}

#[inline(always)]
fn astar_i_in_range_const_cost(
    gh: &Graph<u32, f64, Directed>,
    i_start: usize,
    len: usize,
    target: NodeIndex,
) -> ListChunked {
    let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new("", len, 8, DataType::UInt32);
    for i in i_start..i_start + len {
        let ii = NodeIndex::new(i);
        match astar(gh, ii, |idx| idx == target, |_| 1_u32, |_| 0_u32) {
            Some((_, path)) => {
                let steps = path
                    .into_iter()
                    .skip(1)
                    .map(|n| *gh.node_weight(n).unwrap())
                    .collect_vec();
                builder.append_slice(&steps);
            }
            None => builder.append_null(),
        }
    }
    builder.finish()
}

// This assumes the graph is constructed with weights! See how graphs are constructed in mod.rs
#[inline(always)]
fn astar_i_in_range(
    gh: &Graph<u32, f64, Directed>,
    i_start: usize,
    len: usize,
    target: NodeIndex,
) -> (ListChunked, Float64Chunked) {
    let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new("", len, 8, DataType::UInt32);
    let mut cost_builder: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("", len);

    for i in i_start..i_start + len {
        let ii = NodeIndex::new(i);
        match astar(
            gh,
            ii,
            |idx| idx == target,
            |e| e.weight().clone(),
            |_| 0_f64,
        ) {
            Some((c, path)) => {
                let steps = path
                    .into_iter()
                    .skip(1)
                    .map(|n| *gh.node_weight(n).unwrap())
                    .collect_vec();
                builder.append_slice(&steps);
                cost_builder.append_value(c);
            }
            None => {
                builder.append_null();
                cost_builder.append_null();
            }
        }
    }
    (builder.finish(), cost_builder.finish())
}

#[polars_expr(output_type_func=shortest_path_const_cost_output)]
fn pl_shortest_path_const_cost(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let node_id_col_name = inputs[0].name();
    let gh = super::create_graph(&inputs[..2])?;
    let target = inputs[2].u32()?;
    let target = target.get(0).unwrap();
    let mut target_idx: Option<NodeIndex> = None;
    let mut nodes: Vec<u32> = vec![0u32; gh.node_count()];
    for (idx, w) in gh.node_references() {
        if *w == target {
            target_idx = Some(idx);
        }
        let i = idx.index();
        nodes[i] = *w;
    }
    if target_idx.is_none() {
        return Err(PolarsError::ComputeError(
            "Graph: target is not a valid node identifier.".into(),
        ));
    }
    let target_idx = target_idx.unwrap();
    let nrows = gh.node_count();
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap_or(false);
    let can_parallel = parallel && !context.parallel();

    let ca = if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let out = astar_i_in_range_const_cost(&gh, offset, len, target_idx);
            out.downcast_iter().cloned().collect::<Vec<_>>()
        });

        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        ListChunked::from_chunk_iter("path", chunks.into_iter().flatten())
    } else {
        astar_i_in_range_const_cost(&gh, 0, nrows, target_idx)
    };

    let s1 = Series::from_vec(node_id_col_name, nodes);
    let s2 = ca.with_name("path").into_series();
    let out = StructChunked::new("shortest_path", &[s1, s2])?;
    Ok(out.into_series())
}

#[polars_expr(output_type_func=shortest_path_output)]
fn pl_shortest_path(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let node_id_col_name = inputs[0].name();
    let gh = super::create_graph(&inputs[..3])?;
    let target = inputs[3].u32()?;
    let target = target.get(0).unwrap();
    let mut target_idx: Option<NodeIndex> = None;
    let mut nodes: Vec<u32> = vec![0u32; gh.node_count()];
    for (idx, w) in gh.node_references() {
        if *w == target {
            target_idx = Some(idx);
        }
        let i = idx.index();
        nodes[i] = *w;
    }
    if target_idx.is_none() {
        return Err(PolarsError::ComputeError(
            "Graph: target is not a valid node identifier.".into(),
        ));
    }
    let target_idx = target_idx.unwrap();
    let nrows = gh.node_count();
    let parallel = inputs[4].bool()?;
    let parallel = parallel.get(0).unwrap_or(false);
    let can_parallel = parallel && !context.parallel();

    let (ca1, ca2) = if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        POOL.install(|| {
            let chunks: (Vec<_>, Vec<_>) = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let out = astar_i_in_range(&gh, offset, len, target_idx);
                    (
                        out.0.downcast_iter().cloned().collect::<Vec<_>>(),
                        out.1.downcast_iter().cloned().collect::<Vec<_>>(),
                    )
                })
                .collect();

            (
                ListChunked::from_chunk_iter("path", chunks.0.into_iter().flatten()),
                Float64Chunked::from_chunk_iter("cost", chunks.1.into_iter().flatten()),
            )
        })
    } else {
        astar_i_in_range(&gh, 0, nrows, target_idx)
    };

    let s1 = Series::from_vec(node_id_col_name, nodes);
    let s2 = ca1.with_name("path").into_series();
    let s3 = ca2.with_name("cost").into_series();

    let out = StructChunked::new("shortest_path", &[s1, s2, s3])?;
    Ok(out.into_series())
}

#[polars_expr(output_type_func=dijkstra_output)]
fn pl_shortest_path_dijkstra(inputs: &[Series]) -> PolarsResult<Series> {
    let node_id_col_name = inputs[0].name();
    let target = inputs[2].u32()?;
    let target = target.get(0).unwrap();

    let gh = super::create_graph(&inputs[..2])?;
    let target_idx = gh.node_references().find(|(_, n)| **n == target);
    if target_idx.is_none() {
        return Err(PolarsError::ComputeError(
            "Graph: target is not a valid node identifier.".into(),
        ));
    }

    let target_idx = target_idx.unwrap().0;

    let mut node: Vec<u32> = vec![0; gh.node_count()];
    let mut out: Vec<bool> = vec![false; gh.node_count()];
    let mut out_steps: Vec<u32> = vec![0; gh.node_count()];
    let results = dijkstra(&gh, target_idx, None, |_| 1_u32);
    for (idx, n) in gh.node_references() {
        let i = idx.index();
        if let Some(s) = results.get(&idx) {
            out[i] = true;
            out_steps[i] = *s;
        }
        node[i] = *n;
    }

    let s1 = Series::from_vec(node_id_col_name, node);
    let s2 = BooleanChunked::from_slice("reachable", &out);
    let s2 = s2.into_series();
    let s3 = UInt32Chunked::from_slice("steps", &out_steps);
    let s3 = s3.into_series();

    let out = StructChunked::new("reachable", &[s1, s2, s3])?;
    Ok(out.into_series())
}
