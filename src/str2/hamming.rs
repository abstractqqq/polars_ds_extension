use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use rapidfuzz::distance::hamming;

#[polars_expr(output_type=UInt32)]
fn pl_hamming(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let pad = inputs[2].bool()?;
    let pad = pad.get(0).unwrap_or(false);
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = hamming::BatchComparator::new(r.chars());
        let out: UInt32Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| {
                    let s = op_s?;
                    Some(
                        batched.distance_with_args(s.chars(), &hamming::Args::default().pad(pad))
                            as u32,
                    )
                })
                .collect()
        } else {
            ca1.apply_nonnull_values_generic(DataType::UInt32, |s| {
                batched.distance_with_args(s.chars(), &hamming::Args::default().pad(pad)) as u32
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| {
                    let (w1, w2) = (op_w1?, op_w2?);
                    Some(hamming::distance_with_args(
                        w1.chars(),
                        w2.chars(),
                        &hamming::Args::default().pad(pad),
                    ) as u32)
                })
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| {
                hamming::distance_with_args(
                    x.chars(),
                    y.chars(),
                    &hamming::Args::default().pad(pad),
                ) as u32
            })
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=Boolean)]
fn pl_hamming_filter(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let bound = inputs[2].u32()?;
    let bound = bound.get(0).unwrap() as usize;
    let pad = inputs[3].bool()?;
    let pad = pad.get(0).unwrap_or(false);
    let parallel = inputs[4].bool()?;
    let parallel = parallel.get(0).unwrap();

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = hamming::BatchComparator::new(r.chars());
        let out: BooleanChunked = if parallel {
            ca1.par_iter()
                .map(|op_s| {
                    let s = op_s?;
                    Some(
                        batched
                            .distance_with_args(
                                s.chars(),
                                &hamming::Args::default().pad(pad).score_cutoff(bound),
                            )
                            .is_some(),
                    )
                })
                .collect()
        } else {
            ca1.apply_nonnull_values_generic(DataType::Boolean, |s| {
                batched
                    .distance_with_args(
                        s.chars(),
                        &hamming::Args::default().pad(pad).score_cutoff(bound),
                    )
                    .is_some()
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: BooleanChunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| {
                    let (w1, w2) = (op_w1?, op_w2?);
                    Some(
                        hamming::distance_with_args(
                            w1.chars(),
                            w2.chars(),
                            &hamming::Args::default().pad(pad).score_cutoff(bound),
                        )
                        .is_some(),
                    )
                })
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| {
                hamming::distance_with_args(
                    x.chars(),
                    y.chars(),
                    &hamming::Args::default().pad(pad).score_cutoff(bound),
                )
                .is_some()
            })
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
