use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use rapidfuzz::distance::hamming;

#[inline]
fn hamming(a: &str, b: &str, pad: bool) -> Option<u32> {
    match hamming::distance(a.chars(), b.chars(), pad, None, None) {
        Ok(d) => d.map_or(None, |u| Some(u as u32)),
        _ => None,
    }
}

#[polars_expr(output_type=UInt32)]
fn pl_hamming(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let pad = inputs[2].bool()?;
    let pad = pad.get(0).unwrap_or(false);
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = hamming::BatchComparator::new(r.chars(), pad);
        let out: UInt32Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| {
                    let s = op_s?;
                    batched
                        .distance(s.chars(), None, None)
                        .map_or(None, |op_u| op_u.map_or(None, |u| Some(u as u32)))
                })
                .collect()
        } else {
            ca1.apply_generic(|op_s| {
                let s = op_s?;
                batched
                    .distance(s.chars(), None, None)
                    .map_or(None, |op_u| op_u.map_or(None, |u| Some(u as u32)))
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| {
                    let (w1, w2) = (op_w1?, op_w2?);
                    hamming(w1, w2, pad)
                })
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| hamming(x, y, pad))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
