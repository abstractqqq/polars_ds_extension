// Need to re-think this module because most graph stuff is very expensive (way more than knn),
// And graph queries rely heavily on different data structures
// And building graphs isn't so easy.

mod degree;
mod eigen_centrality;
mod shortest_path;

use petgraph::{Directed, Graph};
use polars::{chunked_array::ops::ChunkUnique, datatypes::{Float64Chunked, UInt32Chunked}, error::{PolarsError, PolarsResult}, series::Series};

fn create_graph_from_nodes(
    nodes: &UInt32Chunked,
    connections: &UInt32Chunked,
    weights: Option<&Float64Chunked>
 ) -> PolarsResult<Graph<u32, f64, Directed>> {

    let mut valid = nodes.len() == connections.len();
    let op_cost = if weights.is_some() {
        let cost = weights.unwrap();
        valid = valid && cost.len() == nodes.len();
        Some(cost)
    } else {
        None
    };
    let node_cnt = nodes.n_unique()?;
    if !valid || node_cnt < 2 {
        return Err(PolarsError::ComputeError(
            "Graph: node, connections, and weight (if exists) columns must have the same length, and there
            must be more than 1 distinct node.".into(),
        ))
    }
    let gh = if let Some(cost) = op_cost {
        Graph::from_edges(
            nodes.into_iter().zip(connections.into_iter()).zip(cost.into_iter())
            .filter(|((i, j), w)| i.is_some() && j.is_some() && w.is_some())
            .map(|((i,j), w)| (i.unwrap(), j.unwrap(), w.unwrap()))
        )
    } else {
        Graph::from_edges(
            nodes.into_iter().zip(connections.into_iter())
            .filter(|(i, j)| i.is_some() && j.is_some())
            .map(|(i,j)| (i.unwrap(), j.unwrap(), 1f64))
        )
    };
    Ok(gh)
}

/// Create the graph, dispatch the creation method
/// based on input column types and number.
pub fn create_graph(
    inputs:&[Series]
) -> PolarsResult<Graph<u32, f64, Directed>> {

    let s1 = inputs[0].u32()?;
    let s2 = inputs[1].u32()?;
    if inputs.len() == 2 {
        create_graph_from_nodes(s1, s2, None)
    } else if inputs.len() == 3 {
        let s3 = inputs[2].f64()?;
        create_graph_from_nodes(s1, s2, Some(s3))
    } else {
        Err(PolarsError::ComputeError(
            "Graph: too many/few inputs.".into(),
        ))
    }
}