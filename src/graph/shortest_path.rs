use crate::utils::{list_u64_output, split_offsets};
use itertools::Itertools;
use petgraph::algo::{astar, dijkstra};
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
    let reachable = Field::new("reachable", DataType::Boolean);
    let cost = Field::new("steps", DataType::UInt32);
    let v = vec![reachable, cost];
    Ok(Field::new("", DataType::Struct(v)))
}

pub fn shortest_path_output(_: &[Field]) -> PolarsResult<Field> {
    let path = Field::new("nodes", DataType::List(Box::new(DataType::UInt64)));
    let cost = Field::new("cost", DataType::Float64);
    let v = vec![path, cost];
    Ok(Field::new("shortest_path", DataType::Struct(v)))
}

#[inline(always)]
fn astar_i_in_range_const_cost(
    gh: &Graph<(), f64, Directed>,
    i_start: usize,
    i_end: usize,
    target: NodeIndex,
    nrows: usize,
) -> ListChunked {
    let mut builder =
        ListPrimitiveChunkedBuilder::<UInt64Type>::new("", nrows, 8, DataType::UInt64);
    for i in i_start..i_end {
        let ii = NodeIndex::new(i);
        match astar(gh, ii, |idx| idx == target, |_| 1_u32, |_| 0_u32) {
            Some((_, path)) => {
                let steps = path
                    .into_iter()
                    .skip(1)
                    .map(|n| n.index() as u64)
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
    gh: &Graph<(), f64, Directed>,
    i_start: usize,
    i_end: usize,
    target: NodeIndex,
    nrows: usize,
) -> (ListChunked, Float64Chunked) {
    let mut builder =
        ListPrimitiveChunkedBuilder::<UInt64Type>::new("", nrows, 8, DataType::UInt64);
    let mut cost_builder: PrimitiveChunkedBuilder<Float64Type> =
        PrimitiveChunkedBuilder::new("", nrows);

    for i in i_start..i_end {
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
                    .map(|n| n.index() as u64)
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

#[polars_expr(output_type_func=list_u64_output)]
fn pl_shortest_path_const_cost(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let edges = inputs[0].list()?;
    let nrows = edges.len();
    let target = inputs[1].u64()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap_or(false);
    let can_parallel = parallel && !context.parallel();

    let gh = super::create_graph(edges, None)?;
    if target.len() == 1 {
        let target = target.get(0).unwrap_or(u64::MAX) as usize;
        if target >= edges.len() {
            return Err(PolarsError::ShapeMismatch(
                "Shortest path: Target index is out of bounds.".into(),
            ));
        }
        let target = NodeIndex::new(target);
        let ca = if can_parallel {
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(nrows, n_threads);
                let chunks: Vec<_> = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let out =
                            astar_i_in_range_const_cost(&gh, offset, offset + len, target, len);
                        out.downcast_iter().cloned().collect::<Vec<_>>()
                    })
                    .collect();
                ListChunked::from_chunk_iter("path", chunks.into_iter().flatten())
            })
        } else {
            astar_i_in_range_const_cost(&gh, 0, nrows, target, nrows)
        };
        Ok(ca.into_series())
    } else {
        Err(PolarsError::ComputeError("Not implemented yet.".into()))
    }
}

#[polars_expr(output_type_func=shortest_path_output)]
fn pl_shortest_path(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let edges = inputs[0].list()?;
    let dist = inputs[1].list()?;
    let nrows = edges.len();
    let target = inputs[2].u64()?;
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap_or(false);
    let can_parallel = parallel && !context.parallel();

    let gh = super::create_graph(edges, Some(dist))?;
    if target.len() == 1 {
        let target = target.get(0).unwrap_or(u64::MAX) as usize;
        if target >= edges.len() {
            return Err(PolarsError::ShapeMismatch(
                "Shortest path: Target index is out of bounds.".into(),
            ));
        }
        let target = NodeIndex::new(target);
        let (ca1, ca2) = if can_parallel {
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(nrows, n_threads);
                let chunks: (Vec<_>, Vec<_>) = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let (path, cost) = astar_i_in_range(&gh, offset, offset + len, target, len);
                        (
                            path.downcast_iter().cloned().collect::<Vec<_>>(),
                            cost.downcast_iter().cloned().collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                let ca1 = ListChunked::from_chunk_iter("path", chunks.0.into_iter().flatten());
                let ca2 = Float64Chunked::from_chunk_iter("cost", chunks.1.into_iter().flatten());
                (ca1, ca2)
            })
        } else {
            astar_i_in_range(&gh, 0, nrows, target, nrows)
        };
        let s1 = ca1.with_name("path").into_series();
        let s2 = ca2.with_name("cost").into_series();
        let out = StructChunked::new("shortest_path", &[s1, s2]).unwrap();
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError("Not implemented yet.".into()))
    }
}

#[polars_expr(output_type_func=dijkstra_output)]
fn pl_shortest_path_dijkstra(inputs: &[Series]) -> PolarsResult<Series> {
    let edges = inputs[0].list()?;
    let target = inputs[1].u64()?;
    let target = target.get(0).unwrap();
    let target = NodeIndex::new(target as usize);

    let gh = super::create_graph(edges, None)?;

    let mut out: Vec<bool> = vec![false; edges.len()];
    let mut out_steps: Vec<u32> = vec![0; edges.len()];
    let results = dijkstra(&gh, target, None, |_| 1_u32);

    for (ni, steps) in results.into_iter() {
        let i = ni.index();
        out[i] = true;
        out_steps[i] = steps;
    }

    let s1 = BooleanChunked::from_slice("reachable", &out);
    let s1 = s1.into_series();
    let s2 = UInt32Chunked::from_slice("steps", &out_steps);
    let s2 = s2.into_series();

    let out = StructChunked::new("", &[s1, s2])?;
    Ok(out.into_series())
}
