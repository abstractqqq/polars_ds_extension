use super::str_set_sim_helper;
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};

fn overlap_coeff(w1: &str, w2: &str, n: usize) -> f64 {
    let (s1, s2, intersect) = str_set_sim_helper(w1, w2, n);
    (intersect as f64) / ((s1.min(s2)) as f64)
}

fn optional_overlap_coeff(op_w1: Option<&str>, op_w2: Option<&str>, n: usize) -> Option<f64> {
    if let (Some(w1), Some(w2)) = (op_w1, op_w2) {
        Some(overlap_coeff(w1, w2, n))
    } else {
        None
    }
}

#[polars_expr(output_type=Float64)]
fn pl_overlap_coeff(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;

    // gauranteed to have 4 input series by the input from Python side.
    // The 3rd input is size of substring length
    let n = inputs[2].u32()?;
    let n = n.get(0).unwrap() as usize;
    // parallel
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();

    if ca2.len() == 1 {
        let r = ca2.get(0); // .unwrap();
        let out: Float64Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| optional_overlap_coeff(op_s, r, n))
                .collect()
        } else {
            ca1.apply_generic(|op_s| optional_overlap_coeff(op_s, r, n))
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_overlap_coeff(op_w1, op_w2, n))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| overlap_coeff(x, y, n))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
