use pathfinding::directed::dijkstra;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

pub fn shortest_path_output(_: &[Field]) -> PolarsResult<Field> {
    let path = Field::new("path", DataType::List(Box::new(DataType::UInt64)));
    let cost = Field::new("cost", DataType::Float64);
    let v = vec![path, cost];
    Ok(Field::new("shortest_path", DataType::Struct(v)))
}

#[polars_expr(output_type_func=shortest_path_output)]
fn pl_shortest_path_const_cost(inputs: &[Series]) -> PolarsResult<Series> {
    let edges = inputs[0].list()?;
    let target = inputs[1].u64()?;
    if target.len() == 1 {
        let target = target.get(0).unwrap();
        if target as usize >= edges.len() {
            return Err(PolarsError::ShapeMismatch(
                "Shortest path: Target index is out of bounds.".into(),
            ));
        }
        // constant, just use cost = 1
        // Construct the graph
        let mut graph: Vec<Vec<u64>> = Vec::with_capacity(edges.len());
        for op_e in edges.into_iter() {
            if let Some(op_e) = op_e {
                let nb = op_e.u64()?;
                let nb = nb.into_no_null_iter().collect::<Vec<u64>>();
                graph.push(nb);
            }
        }
        let mut path_builder: ListPrimitiveChunkedBuilder<UInt64Type> =
            ListPrimitiveChunkedBuilder::new("path", edges.len(), 8, DataType::UInt64);

        let mut cost_builder: PrimitiveChunkedBuilder<Float64Type> =
            PrimitiveChunkedBuilder::new("cost", edges.len());

        for i in 0..edges.len() as u64 {
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
                    path_builder.append_slice(&path);
                    cost_builder.append_value(path.len() as f64 - 1.0); // path contains self
                }
                None => {
                    path_builder.append_null();
                    cost_builder.append_null();
                }
            }
        }

        let path = path_builder.finish();
        let path = path.into_series();
        let cost = cost_builder.finish();
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
