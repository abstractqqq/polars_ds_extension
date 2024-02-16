use petgraph::{
    stable_graph::NodeIndex,
    visit::GetAdjacencyMatrix,
    Direction::{Incoming, Outgoing},
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

pub fn in_out_deg_output(_: &[Field]) -> PolarsResult<Field> {
    let in_deg = Field::new("in_deg", DataType::UInt32);
    let out_deg = Field::new("out_deg", DataType::UInt32);
    let v = vec![in_deg, out_deg];
    Ok(Field::new("in_out_deg", DataType::Struct(v)))
}

#[polars_expr(output_type=UInt32)]
fn pl_graph_deg(inputs: &[Series]) -> PolarsResult<Series> {
    let edges = inputs[0].list()?;
    let gh = super::create_graph(edges, None)?;
    let ca = UInt32Chunked::from_iter_values(
        "deg",
        (0..edges.len()).map(|i| {
            let idx = NodeIndex::new(i);
            gh.neighbors(idx).count() as u32
        }),
    );
    Ok(ca.into_series())
}

#[polars_expr(output_type_func=in_out_deg_output)]
fn pl_graph_in_out_deg(inputs: &[Series]) -> PolarsResult<Series> {
    let edges = inputs[0].list()?;
    let gh = super::create_graph(edges, None)?;
    let mut ins: Vec<u32> = Vec::with_capacity(edges.len());
    let mut out: Vec<u32> = Vec::with_capacity(edges.len());
    for i in 0..edges.len() {
        let idx = NodeIndex::new(i);
        ins.push(gh.neighbors_directed(idx, Incoming).count() as u32);
        out.push(gh.neighbors_directed(idx, Outgoing).count() as u32);
    }
    let s1 = Series::from_vec("in_deg", ins);
    let s2 = Series::from_vec("out_deg", out);
    let out = StructChunked::new("deg", &[s1, s2])?;
    Ok(out.into_series())
}
