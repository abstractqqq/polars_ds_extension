use crate::utils::split_offsets;
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use rapidfuzz::distance::hamming;

#[inline(always)]
fn hamming_within_bound(x: &str, y: &str, bound: usize) -> bool {
    let can_compare = hamming::distance_with_args(
        x.chars(),
        y.chars(),
        &hamming::Args::default().score_cutoff(bound),
    );
    if can_compare.is_ok() {
        can_compare.unwrap().is_some()
    } else {
        false
    }
}

#[polars_expr(output_type=UInt32)]
fn pl_hamming(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = hamming::BatchComparator::new(r.chars());
        let out: UInt32Chunked = if can_parallel {
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(ca1.len(), n_threads);
                let chunks: Vec<_> = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let s1 = ca1.slice(offset as i64, len);
                        let out: UInt32Chunked = s1.apply_generic(|op_s| {
                            let s = op_s?;
                            match batched.distance(s.chars()) {
                                Ok(d) => Some(d as u32),
                                Err(_) => None,
                            }
                        });
                        out.downcast_iter().cloned().collect::<Vec<_>>()
                    })
                    .collect();
                UInt32Chunked::from_chunk_iter(ca1.name(), chunks.into_iter().flatten())
            })
        } else {
            ca1.apply_generic(|op_s| {
                let s = op_s?;
                match batched.distance(s.chars()) {
                    Ok(d) => Some(d as u32),
                    Err(_) => None,
                }
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if can_parallel {
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(ca1.len(), n_threads);
                let chunks: Vec<_> = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let s1 = ca1.slice(offset as i64, len);
                        let s2 = ca2.slice(offset as i64, len);
                        let out: UInt32Chunked = s1
                            .into_iter()
                            .zip(s2.into_iter())
                            .map(|(op_s1, op_s2)| {
                                if let (Some(s1), Some(s2)) = (op_s1, op_s2) {
                                    match hamming::distance(s1.chars(), s2.chars()) {
                                        Ok(d) => Some(d as u32),
                                        Err(_) => None,
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        out.downcast_iter().cloned().collect::<Vec<_>>()
                    })
                    .collect();
                UInt32Chunked::from_chunk_iter(ca1.name(), chunks.into_iter().flatten())
            })
        } else {
            ca1.into_iter()
                .zip(ca2.into_iter())
                .map(|(op_s1, op_s2)| {
                    if let (Some(s1), Some(s2)) = (op_s1, op_s2) {
                        match hamming::distance(s1.chars(), s2.chars()) {
                            Ok(d) => Some(d as u32),
                            Err(_) => None,
                        }
                    } else {
                        None
                    }
                })
                .collect()
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=UInt32)]
fn pl_hamming_padded(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = hamming::BatchComparator::new(r.chars());
        let out: UInt32Chunked = if can_parallel {
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(ca1.len(), n_threads);
                let chunks: Vec<_> = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let s1 = ca1.slice(offset as i64, len);
                        let out: UInt32Chunked =
                            s1.apply_nonnull_values_generic(DataType::UInt32, |s| {
                                batched.distance_with_args(
                                    s.chars(),
                                    &hamming::Args::default().pad(true),
                                ) as u32
                            });
                        out.downcast_iter().cloned().collect::<Vec<_>>()
                    })
                    .collect();
                UInt32Chunked::from_chunk_iter(ca1.name(), chunks.into_iter().flatten())
            })
        } else {
            ca1.apply_nonnull_values_generic(DataType::UInt32, |s| {
                batched.distance_with_args(s.chars(), &hamming::Args::default().pad(true)) as u32
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if can_parallel {
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(ca1.len(), n_threads);
                let chunks: Vec<_> = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let s1 = ca1.slice(offset as i64, len);
                        let s2 = ca2.slice(offset as i64, len);
                        let out: UInt32Chunked = binary_elementwise_values(&s1, &s2, |x, y| {
                            hamming::distance_with_args(
                                x.chars(),
                                y.chars(),
                                &hamming::Args::default().pad(true),
                            ) as u32
                        });
                        out.downcast_iter().cloned().collect::<Vec<_>>()
                    })
                    .collect();
                UInt32Chunked::from_chunk_iter(ca1.name(), chunks.into_iter().flatten())
            })
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| {
                hamming::distance_with_args(
                    x.chars(),
                    y.chars(),
                    &hamming::Args::default().pad(true),
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
fn pl_hamming_filter(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let bound = inputs[2].u32()?;
    let bound = bound.get(0).unwrap() as usize;
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = hamming::BatchComparator::new(r.chars());
        let out: BooleanChunked = if can_parallel {
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(ca1.len(), n_threads);
                let chunks: Vec<_> = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let s1 = ca1.slice(offset as i64, len);
                        let out: BooleanChunked =
                            s1.apply_nonnull_values_generic(DataType::Boolean, |s| {
                                let can_compare = batched.distance_with_args(
                                    s.chars(),
                                    &hamming::Args::default().score_cutoff(bound),
                                );
                                if can_compare.is_ok() {
                                    can_compare.unwrap().is_some()
                                } else {
                                    false
                                }
                            });
                        out.downcast_iter().cloned().collect::<Vec<_>>()
                    })
                    .collect();
                BooleanChunked::from_chunk_iter(ca1.name(), chunks.into_iter().flatten())
            })
        } else {
            ca1.apply_nonnull_values_generic(DataType::Boolean, |s| {
                let can_compare = batched
                    .distance_with_args(s.chars(), &hamming::Args::default().score_cutoff(bound));
                if can_compare.is_ok() {
                    can_compare.unwrap().is_some()
                } else {
                    false
                }
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: BooleanChunked = if can_parallel {
            POOL.install(|| {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(ca1.len(), n_threads);
                let chunks: Vec<_> = splits
                    .into_par_iter()
                    .map(|(offset, len)| {
                        let s1 = ca1.slice(offset as i64, len);
                        let s2 = ca2.slice(offset as i64, len);
                        let out: BooleanChunked = binary_elementwise_values(&s1, &s2, |x, y| {
                            hamming_within_bound(x, y, bound)
                        });
                        out.downcast_iter().cloned().collect::<Vec<_>>()
                    })
                    .collect();
                BooleanChunked::from_chunk_iter(ca1.name(), chunks.into_iter().flatten())
            })
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| hamming_within_bound(x, y, bound))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
