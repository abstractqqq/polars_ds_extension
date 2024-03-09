// Need to re-think this module because most graph stuff is very expensive (way more than knn),
// And graph queries rely heavily on different data structures
// And building graphs isn't so easy.

mod degree;
mod eigen_centrality;
mod shortest_path;

use hashbrown::{HashMap, HashSet};
use petgraph::{adj::NodeIndex, csr::IndexType, Directed, Graph};
use polars::{
    chunked_array::ops::ChunkUnique,
    datatypes::{Float64Chunked, UInt32Chunked},
    error::{PolarsError, PolarsResult},
    series::Series,
};

fn create_graph_from_nodes(
    nodes: &UInt32Chunked,
    connections: &UInt32Chunked,
    weights: Option<&Float64Chunked>,
) -> PolarsResult<Graph<u32, f64, Directed>> {
    let mut valid = nodes.len() == connections.len();
    let op_cost = if weights.is_some() {
        let cost = weights.unwrap();
        valid = valid && cost.len() == nodes.len();
        Some(cost)
    } else {
        None
    };

    let u1 = nodes.unique()?;
    let u2 = connections.unique()?;
    let mut temp: HashSet<u32> = HashSet::new();
    temp.extend(u1.into_no_null_iter());
    temp.extend(u2.into_no_null_iter());

    let node_cnt = temp.len();
    if !valid || node_cnt < 2 {
        return Err(PolarsError::ComputeError(
            "Graph: node, connections, and weight (if exists) columns must have the same length, and there
            must be more than 1 distinct node.".into(),
        ));
    }

    let mut gh: Graph<u32, f64> = Graph::new();
    let mut mapper: HashMap<u32, usize> = HashMap::new();
    for node in temp.into_iter() {
        let idx = gh.add_node(node);
        mapper.insert(node, idx.index());
    }

    if let Some(cost) = op_cost {
        for ((ii, jj), ww) in nodes
            .into_iter()
            .zip(connections.into_iter())
            .zip(cost.into_iter())
        {
            if let (Some(i), Some(j), Some(w)) = (ii, jj, ww) {
                let idx1 = mapper.get(&i).unwrap();
                let idx2 = mapper.get(&j).unwrap();
                let idx1 = NodeIndex::new(*idx1);
                let idx2 = NodeIndex::new(*idx2);
                gh.add_edge(idx1, idx2, w);
            }
        }
    } else {
        for (ii, jj) in nodes.into_iter().zip(connections.into_iter()) {
            if let (Some(i), Some(j)) = (ii, jj) {
                let idx1 = mapper.get(&i).unwrap();
                let idx2 = mapper.get(&j).unwrap();
                let idx1 = NodeIndex::new(*idx1);
                let idx2 = NodeIndex::new(*idx2);
                gh.add_edge(idx1, idx2, 1f64);
            }
        }
    };
    Ok(gh)
}

/// Create the graph, dispatch the creation method
/// based on input column types and number.
pub fn create_graph(inputs: &[Series]) -> PolarsResult<Graph<u32, f64, Directed>> {
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
