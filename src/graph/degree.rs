use petgraph::{
    visit::IntoNodeReferences,
    Direction::{Incoming, Outgoing},
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

pub fn deg_output(_: &[Field]) -> PolarsResult<Field> {
    let node = Field::new("node", DataType::UInt32);
    let deg = Field::new("deg", DataType::UInt32);
    let v = vec![node, deg];
    Ok(Field::new("degree", DataType::Struct(v)))
}

pub fn in_out_deg_output(_: &[Field]) -> PolarsResult<Field> {
    let node = Field::new("node", DataType::UInt32);
    let in_deg = Field::new("in", DataType::UInt32);
    let out_deg = Field::new("out", DataType::UInt32);
    let v = vec![node, in_deg, out_deg];
    Ok(Field::new("degree", DataType::Struct(v)))
}

#[polars_expr(output_type_func=deg_output)]
fn pl_graph_deg(inputs: &[Series]) -> PolarsResult<Series> {
    let gh = super::create_graph(inputs)?;
    let mut nodes: Vec<u32> = Vec::with_capacity(gh.node_count());
    let mut degrees: Vec<u32> = Vec::with_capacity(gh.node_count());

    for (id, n) in gh.node_references() {
        nodes.push(*n);
        degrees.push(gh.neighbors(id).count() as u32)
    }
    let s1 = Series::from_vec("node", nodes);
    let s2 = Series::from_vec("deg", degrees);
    let out = StructChunked::new("degree", &[s1, s2])?;
    Ok(out.into_series())
}

#[polars_expr(output_type_func=in_out_deg_output)]
fn pl_graph_in_out_deg(inputs: &[Series]) -> PolarsResult<Series> {
    let gh = super::create_graph(inputs)?;
    let mut nodes: Vec<u32> = Vec::with_capacity(gh.node_count());
    let mut ins: Vec<u32> = Vec::with_capacity(gh.node_count());
    let mut out: Vec<u32> = Vec::with_capacity(gh.node_count());
    for (id, n) in gh.node_references() {
        nodes.push(*n);
        ins.push(gh.neighbors_directed(id, Incoming).count() as u32);
        out.push(gh.neighbors_directed(id, Outgoing).count() as u32);
    }
    let s1 = Series::from_vec("node", nodes);
    let s2 = Series::from_vec("in", ins);
    let s3 = Series::from_vec("out", out);
    let out = StructChunked::new("degree", &[s1, s2, s3])?;
    Ok(out.into_series())
}
