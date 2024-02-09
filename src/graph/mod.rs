// Need to re-think this module because most graph stuff is very expensive (way more than knn),
// And graph queries rely heavily on different data structures
// And building graphs isn't so easy.

mod degree;
mod eigen_centrality;
mod shortest_path;

use petgraph::{stable_graph::NodeIndex, Directed, Graph};
use polars::{
    datatypes::ListChunked,
    error::{PolarsError, PolarsResult},
};

// Here internally I am using an edge's weight to represent the cost.
// Graph<usize, f64, Undirected>, where usize is Node weight, which is unused (actually same as index)
// And f64 is edge weight.

pub fn create_graph(
    edges: &ListChunked,
    edge_cost: Option<&ListChunked>,
) -> PolarsResult<Graph<usize, f64, Directed>> {
    let mut gh = Graph::with_capacity(edges.len(), 8);
    for i in 0..edges.len() {
        gh.add_node(i);
    }
    if let Some(cost) = edge_cost {
        let it = (edges.into_iter().zip(cost.into_iter())).enumerate();
        for (a, (op_e, op_c)) in it {
            let i = NodeIndex::new(a);
            if let (Some(e), Some(c)) = (op_e, op_c) {
                let neighbors = e.u64()?;
                let dist = c.f64()?;
                if neighbors.len() != dist.len()
                    || neighbors.null_count() > 0
                    || dist.null_count() > 0
                {
                    return Err(PolarsError::ComputeError(
                        "Edge and cost list must have the same length for 
                    all rows, and both cannot contain nulls."
                            .into(),
                    ));
                }
                for (b, d) in neighbors.into_no_null_iter().zip(dist.into_no_null_iter()) {
                    let j = NodeIndex::new(b as usize);
                    gh.add_edge(i, j, d);
                }
            }
        }
    } else {
        for (a, op_e) in edges.into_iter().enumerate() {
            let i = NodeIndex::new(a);
            if let Some(e) = op_e {
                let neighbors = e.u64()?;
                for b in neighbors.into_no_null_iter() {
                    let j = NodeIndex::new(b as usize);
                    gh.add_edge(i, j, 1_f64);
                }
            }
        }
    }
    Ok(gh)
}
