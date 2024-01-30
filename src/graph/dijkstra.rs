use crate::list_u64_output;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use pathfinding::directed::dijkstra;
use polars::prelude::*;
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::iter::{
        FromParallelIterator, IndexedParallelIterator, IntoParallelIterator, ParallelBridge,
        ParallelIterator,
    },
};

// These are very ad-hoc functions that are used at least twice. They serve no other
// purpose than making the code cleaner.
#[inline(always)]
fn _u64_vec(op_s: Option<Series>) -> Vec<u64> {
    if let Some(s) = op_s {
        match s.u64() {
            Ok(u) => u.into_no_null_iter().collect(),
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    }
}

#[inline(always)]
fn _u64_f64_vec(opt: (Option<Series>, Option<Series>)) -> Vec<(u64, f64)> {
    if let (Some(s), Some(c)) = opt {
        match (s.u64(), c.f64()) {
            (Ok(s), Ok(d)) => {
                if s.len() != d.len() || s.null_count() > 0 || d.null_count() > 0 {
                    Vec::new()
                } else {
                    // Safe
                    s.into_iter()
                        .zip(d.into_iter())
                        .map(|(uu, dd)| (uu.unwrap(), dd.unwrap()))
                        .collect()
                }
            }
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    }
}

#[inline(always)]
fn _dijkstra_const(graph: &Vec<Vec<u64>>, i: u64, target: u64) -> Option<Series> {
    let result = dijkstra::dijkstra(
        &i,
        |i| {
            graph[*i as usize]
                .iter()
                .map(|j| (*j, 1_usize))
                .collect::<Vec<_>>()
        },
        |i| *i == target,
    );
    match result {
        Some((mut path, _)) => {
            path.shrink_to_fit();
            Some(Series::from_vec("", path))
        }
        None => None,
    }
}

#[inline(always)]
fn _dijkstra_w_cost(graph: &Vec<Vec<(u64, f64)>>, i: u64, target: u64) -> Option<(Vec<u64>, f64)> {
    let result = dijkstra::dijkstra(
        &i,
        |i| {
            graph[*i as usize]
                .iter()
                .map(|tup| (tup.0, OrderedFloat::from(tup.1)))
                .collect::<Vec<_>>()
        },
        |i| *i == target,
    );
    match result {
        Some((mut path, c)) => {
            path.shrink_to_fit();
            Some((path, c.into()))
        }
        None => None,
    }
}

#[polars_expr(output_type_func=list_u64_output)]
fn pl_shortest_path_const_cost(inputs: &[Series]) -> PolarsResult<Series> {
    let edges = inputs[0].list()?;
    let target = inputs[1].u64()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap_or(false);
    if target.len() == 1 {
        let target = target.get(0).unwrap();
        if target as usize >= edges.len() {
            return Err(PolarsError::ShapeMismatch(
                "Shortest path: Target index is out of bounds.".into(),
            ));
        }
        // constant, just use cost = 1
        // Construct the graph. Copying, expensive.
        let graph: Vec<Vec<u64>> = if parallel {
            let mut gh: Vec<Vec<u64>> = Vec::with_capacity(edges.len());
            let mut owned_edges = edges.clone(); //cheap
            owned_edges
                .par_iter_indexed()
                .map(_u64_vec)
                .collect_into_vec(&mut gh);
            gh
        } else {
            edges.into_iter().map(_u64_vec).collect_vec()
        };

        let path = if parallel {
            ListChunked::from_par_iter(
                (0..edges.len() as u64)
                    .into_par_iter()
                    .map(|i| _dijkstra_const(&graph, i, target)),
            )
        } else {
            ListChunked::from_iter(
                (0..edges.len() as u64).map(|i| _dijkstra_const(&graph, i, target)),
            )
        };
        let path = path.into_series();
        Ok(path.into_series())
    } else if target.len() == edges.len() {
        Err(PolarsError::ShapeMismatch("Not implemented yet.".into()))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or target must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type_func=list_u64_output)]
fn pl_shortest_path(inputs: &[Series]) -> PolarsResult<Series> {
    let edges = inputs[0].list()?;
    let costs = inputs[1].list()?;
    let target = inputs[2].u64()?;
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap_or(false);
    if target.len() == 1 {
        let target = target.get(0).unwrap();
        if target as usize >= edges.len() {
            return Err(PolarsError::ShapeMismatch(
                "Shortest path: Target index is out of bounds.".into(),
            ));
        }
        // constant, just use cost = 1
        // Construct the graph. Copying, expensive.
        let graph = if parallel {
            let mut gh = Vec::with_capacity(edges.len());
            let mut owned_edges = edges.clone(); //cheap
            let mut owned_costs = costs.clone(); //cheap
            owned_edges
                .par_iter_indexed()
                .zip(owned_costs.par_iter_indexed())
                .map(_u64_f64_vec)
                .collect_into_vec(&mut gh);
            gh
        } else {
            edges
                .into_iter()
                .zip(costs.into_iter())
                .map(_u64_f64_vec)
                .collect_vec()
        };

        let mut path_builder: ListPrimitiveChunkedBuilder<UInt64Type> =
            ListPrimitiveChunkedBuilder::new("path", edges.len(), 8, DataType::UInt64);
        let mut cost_builder: PrimitiveChunkedBuilder<Float64Type> =
            PrimitiveChunkedBuilder::new("cost", edges.len());

        let (path, cost) = if parallel {
            let buffer: Vec<u64> = (0..edges.len() as u64).collect();
            let mut out: Vec<Option<(Vec<u64>, f64)>> = Vec::with_capacity(edges.len());
            buffer
                .into_par_iter()
                .map(|i| _dijkstra_w_cost(&graph, i, target))
                .collect_into_vec(&mut out);
            for opt in out {
                match opt {
                    Some((p, c)) => {
                        path_builder.append_slice(&p);
                        cost_builder.append_value(c);
                    }
                    None => {
                        path_builder.append_null();
                        cost_builder.append_null();
                    }
                }
            }
            (path_builder.finish(), cost_builder.finish())
        } else {
            (0..edges.len() as u64).for_each(|i| match _dijkstra_w_cost(&graph, i, target) {
                Some((p, c)) => {
                    path_builder.append_slice(&p);
                    cost_builder.append_value(c);
                }
                None => {
                    path_builder.append_null();
                    cost_builder.append_null();
                }
            });
            (path_builder.finish(), cost_builder.finish())
        };
        let path = path.into_series();
        let cost = cost.into_series();
        let out = StructChunked::new("shortest_path", &[path, cost])?;
        Ok(out.into_series())
    } else if target.len() == edges.len() {
        Err(PolarsError::ShapeMismatch("Not implemented yet.".into()))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or target must be a scalar.".into(),
        ))
    }
}
