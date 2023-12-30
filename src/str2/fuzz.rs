/// Returns a simple ratio between two strings or `None` if `ratio < score_cutoff`
/// The simple ratio is
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use rapidfuzz::fuzz::{ratio_with_args, Args, RatioBatchComparator};

#[polars_expr(output_type=UInt32)]
fn pl_fuzz(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let cutoff = inputs[2].f64()?;
    let cutoff = cutoff.get(0).unwrap_or(0.);
    let args = Args::default().score_cutoff(cutoff);
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = RatioBatchComparator::new(r.chars());
        let out: Float64Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| {
                    let s = op_s?;
                    batched.similarity_with_args(s.chars(), &args)
                })
                .collect()
        } else {
            ca1.apply_generic(|op_s| {
                let s = op_s?;
                batched.similarity_with_args(s.chars(), &args)
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| {
                    let (w1, w2) = (op_w1?, op_w2?);
                    ratio_with_args(w1.chars(), w2.chars(), &args)
                })
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |w1, w2| {
                ratio_with_args(w1.chars(), w2.chars(), &args)
            })
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
