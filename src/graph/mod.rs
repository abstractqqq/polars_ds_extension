// Need to re-think this module because most graph stuff is very expensive (way more than knn),
// And graph queries rely heavily on different data structures
// And building graphs isn't so easy.

mod degree;
mod eigen_centrality;
mod shortest_path;

use petgraph::{stable_graph::NodeIndex, Directed, Graph};
use polars::{datatypes::ListChunked, error::PolarsResult};

// Here internally I am using an edge's weight to represent the cost.
// Graph<(), f64, Directed>, where f64 is edge weight.

pub fn create_graph(
    edges: &ListChunked,
    edge_cost: Option<&ListChunked>,
) -> PolarsResult<Graph<(), f64, Directed>> {
    let mut gh = Graph::with_capacity(edges.len(), 8);
    let node_list = (0..edges.len())
        .map(|_| gh.add_node(()))
        .collect::<Vec<NodeIndex>>();
    if let Some(cost) = edge_cost {
        let it = node_list
            .into_iter()
            .zip(edges.into_iter().zip(cost.into_iter()));
        for (i, (op_e, op_c)) in it {
            if let (Some(e), Some(c)) = (op_e, op_c) {
                let neighbors = e.u64()?;
                let dist = c.f64()?;
                if neighbors.len() == dist.len()
                    && neighbors.null_count() == 0
                    && dist.null_count() == 0
                {
                    for (b, d) in neighbors.into_no_null_iter().zip(dist.into_no_null_iter()) {
                        let j = NodeIndex::new(b as usize);
                        gh.add_edge(i, j, d);
                    }
                } // else, don't add the edges
            }
        }
    } else {
        for (i, op_e) in node_list.into_iter().zip(edges.into_iter()) {
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
