use crate::utils::{list_str_output, split_offsets};
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use rapidfuzz::distance::{hamming, levenshtein};
use serde::Deserialize;
use std::collections::BinaryHeap;

#[derive(Deserialize, Debug)]
pub(crate) struct KnnStrKwargs {
    pub(crate) k: usize,
    pub(crate) metric: String,
    pub(crate) threshold: usize,
    pub(crate) parallel: bool,
}

// Can we improve performance by removing the function pointers?

fn levenshtein_nearest<'a>(s: &str, cutoff: usize, vocab: &'a StringChunked) -> Option<&'a str> {
    let batched = levenshtein::BatchComparator::new(s.chars());
    // Most similar == having smallest distance
    let mut best: usize = usize::MAX;
    let mut nearest_str: Option<&str> = None;

    for arr in vocab.downcast_iter() {
        for w in arr.values_iter() {
            if let Some(d) = batched.distance_with_args(
                w.chars(),
                &levenshtein::Args::default().score_cutoff(cutoff),
            ) {
                if d < best {
                    best = d;
                    nearest_str = Some(w);
                }
            }
        }
    }
    nearest_str
}

/// Returns a series of strings, so that it does less redundant work later.
/// Only use this internally.
fn levenshtein_knn(s: &str, cutoff: usize, k: usize, vocab: &StringChunked) -> Series {
    let batched = levenshtein::BatchComparator::new(s.chars());
    // Most similar == having smallest distance
    // A binary heap is a max heap. So we do -usize
    // Vocab is gauranteed to contain no null
    let mut heap: BinaryHeap<(isize, &str)> = BinaryHeap::with_capacity(vocab.len());
    for arr in vocab.downcast_iter() {
        for w in arr.values_iter() {
            if let Some(d) = batched.distance_with_args(
                w.chars(),
                &levenshtein::Args::default().score_cutoff(cutoff),
            ) {
                // all distances are negative in heap. md = minus d
                let md = -(d as isize);
                heap.push((md, w));
            }
        }
    }
    let ca: ChunkedArray<StringType> = StringChunked::from_iter_values(
        "",
        heap.into_sorted_vec()
            .into_iter()
            .rev()
            .map(|(_, w)| w)
            .take(k),
    );
    ca.into_series()
}

fn hamming_nearest<'a>(s: &str, cutoff: usize, vocab: &'a StringChunked) -> Option<&'a str> {
    let batched = hamming::BatchComparator::new(s.chars());
    let mut best: usize = usize::MAX;
    let mut nearest_str: Option<&str> = None;

    for arr in vocab.downcast_iter() {
        for w in arr.values_iter() {
            if let Ok(ss) = batched
                .distance_with_args(w.chars(), &hamming::Args::default().score_cutoff(cutoff))
            {
                if let Some(d) = ss {
                    if d < best {
                        best = d;
                        nearest_str = Some(w);
                    }
                }
            }
        }
    }
    nearest_str
}

fn hamming_knn(s: &str, cutoff: usize, k: usize, vocab: &StringChunked) -> Series {
    let batched = hamming::BatchComparator::new(s.chars());
    // Most similar == having smallest distance
    // A binary heap is a max heap. So we do -usize
    // Vocab is gauranteed to contain no null
    let mut heap: BinaryHeap<(isize, &str)> = BinaryHeap::with_capacity(vocab.len());
    for arr in vocab.downcast_iter() {
        for w in arr.values_iter() {
            if let Ok(dd) = batched
                .distance_with_args(w.chars(), &hamming::Args::default().score_cutoff(cutoff))
            {
                if let Some(d) = dd {
                    let md = -(d as isize);
                    heap.push((md, w));
                }
            }
        }
    }
    let ca: ChunkedArray<StringType> = StringChunked::from_iter_values(
        "",
        heap.into_sorted_vec()
            .into_iter()
            .rev()
            .map(|(_, w)| w)
            .take(k),
    );
    ca.into_series()
}

#[polars_expr(output_type=String)]
pub fn pl_nearest_str(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KnnStrKwargs,
) -> PolarsResult<Series> {
    let s = inputs[0].str()?;
    let vocab = inputs[1].str()?;

    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    let cutoff = kwargs.threshold;

    let func = match kwargs.metric.as_str() {
        "hamming" => hamming_nearest,
        _ => levenshtein_nearest,
    };

    if can_parallel {
        let out = POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(s.len(), n_threads);
            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let ca = s.slice(offset as i64, len);
                    let out: StringChunked = ca.apply_generic(|op_s| {
                        let s = op_s?;
                        func(s, cutoff, vocab)
                    });
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            StringChunked::from_chunk_iter(s.name(), chunks.into_iter().flatten())
        });
        Ok(out.into_series())
    } else {
        let out: StringChunked = s.apply_generic(|op_s| {
            let s = op_s?;
            func(s, cutoff, vocab)
        });
        Ok(out.into_series())
    }
}

#[polars_expr(output_type_func=list_str_output)]
pub fn pl_knn_str(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KnnStrKwargs,
) -> PolarsResult<Series> {
    let s = inputs[0].str()?;
    let binding = inputs[1].drop_nulls();
    let vocab = binding.str()?;

    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    let cutoff = kwargs.threshold;
    let k = kwargs.k;
    // This is a function pointer hmm
    let func = match kwargs.metric.as_str() {
        "hamming" => hamming_knn,
        _ => levenshtein_knn,
    };

    let out = if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(s.len(), n_threads);
            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let ca = s.slice(offset as i64, len);
                    let mut builder = ListStringChunkedBuilder::new("", ca.len(), k);
                    for arr in ca.downcast_iter() {
                        for op_s in arr.iter() {
                            match op_s {
                                Some(s) => {
                                    let series = func(s, cutoff, k, vocab);
                                    let _ = builder.append_series(&series);
                                }
                                None => builder.append_null(),
                            }
                        }
                    }
                    let out = builder.finish();
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            ListChunked::from_chunk_iter(s.name(), chunks.into_iter().flatten())
        })
    } else {
        let mut builder = ListStringChunkedBuilder::new("", s.len(), k);
        for arr in s.downcast_iter() {
            for op_w in arr.iter() {
                match op_w {
                    Some(w) => {
                        let neighbors = func(w, cutoff, k, vocab);
                        let _ = builder.append_series(&neighbors);
                    }
                    None => builder.append_null(),
                }
            }
        }
        builder.finish()
    };
    Ok(out.into_series())
}
