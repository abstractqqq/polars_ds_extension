use crate::list_str_output;
use polars::prelude::*;
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::{iter::FromParallelIterator, prelude::ParallelIterator},
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

fn levenshtein_nearest<'a>(s: &str, cutoff: usize, vocab: &'a StringChunked) -> &'a str {
    let batched = levenshtein::BatchComparator::new(s.chars());
    // Most similar == having smallest distance
    let mut best: usize = usize::MAX;
    let mut nearest_str: &str = "";
    vocab.into_iter().for_each(|op_w| {
        if let Some(w) = op_w {
            if let Some(d) = batched.distance_with_args(
                w.chars(),
                &levenshtein::Args::default().score_cutoff(cutoff),
            ) {
                if d < best {
                    best = d;
                    nearest_str = w;
                }
            }
        }
    });
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
    vocab.into_iter().for_each(|op_w| {
        let w = op_w.unwrap();
        if let Some(d) = batched.distance_with_args(
            w.chars(),
            &levenshtein::Args::default().score_cutoff(cutoff),
        ) {
            // all distances are negative in heap. md = minus d
            let md = -(d as isize);
            heap.push((md, w));
        }
    });
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

fn hamming_nearest<'a>(s: &str, cutoff: usize, vocab: &'a StringChunked) -> &'a str {
    let batched = hamming::BatchComparator::new(s.chars());
    let mut best: usize = usize::MAX;
    let mut nearest_str: &str = "";
    vocab.into_iter().for_each(|op_w| {
        if let Some(w) = op_w {
            if let Ok(ss) = batched
                .distance_with_args(w.chars(), &hamming::Args::default().score_cutoff(cutoff))
            {
                if let Some(d) = ss {
                    if d < best {
                        best = d;
                        nearest_str = w;
                    }
                }
            }
        }
    });
    nearest_str
}

fn hamming_knn(s: &str, cutoff: usize, k: usize, vocab: &StringChunked) -> Series {
    let batched = hamming::BatchComparator::new(s.chars());
    // Most similar == having smallest distance
    // A binary heap is a max heap. So we do -usize
    // Vocab is gauranteed to contain no null
    let mut heap: BinaryHeap<(isize, &str)> = BinaryHeap::with_capacity(vocab.len());
    vocab.into_iter().for_each(|op_w| {
        let w = op_w.unwrap();
        if let Ok(dd) =
            batched.distance_with_args(w.chars(), &hamming::Args::default().score_cutoff(cutoff))
        {
            if let Some(d) = dd {
                let md = -(d as isize);
                heap.push((md, w));
            }
        }
    });
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
pub fn pl_nearest_str(inputs: &[Series], kwargs: KnnStrKwargs) -> PolarsResult<Series> {
    let s = inputs[0].str()?;
    let vocab = inputs[1].str()?;

    let parallel = kwargs.parallel;
    let cutoff = kwargs.threshold;
    // This is a function pointer hmm
    let func = match kwargs.metric.as_str() {
        "hamming" => hamming_nearest,
        _ => levenshtein_nearest,
    };

    if parallel {
        let out_par_iter = s.par_iter().map(|op_s| {
            let s = op_s?;
            Some(func(s, cutoff, vocab))
        });
        let out = StringChunked::from_par_iter(out_par_iter);
        Ok(out.into_series())
    } else {
        let out = s.apply_values(|s| func(s, cutoff, vocab).into());
        Ok(out.into_series())
    }
}

#[polars_expr(output_type_func=list_str_output)]
pub fn pl_knn_str(inputs: &[Series], kwargs: KnnStrKwargs) -> PolarsResult<Series> {
    let s = inputs[0].str()?;
    let binding = inputs[1].drop_nulls();
    let vocab = binding.str()?;

    let parallel = kwargs.parallel;
    let cutoff = kwargs.threshold;
    let k = kwargs.k;
    // This is a function pointer hmm
    let func = match kwargs.metric.as_str() {
        "hamming" => hamming_knn,
        _ => levenshtein_knn,
    };

    let out = if parallel {
        let out_par_iter = s.par_iter().map(|op_w| {
            let w = op_w?;
            Some(func(w, cutoff, k, vocab))
        });
        ListChunked::from_par_iter(out_par_iter)
    } else {
        let mut builder = ListStringChunkedBuilder::new("", s.len(), k);
        for op_s in s.into_iter() {
            if let Some(w) = op_s {
                let neighbors = func(w, cutoff, k, vocab);
                let _ = builder.append_series(&neighbors);
            } else {
                builder.append_null()
            }
        }
        builder.finish()
    };
    Ok(out.into_series())
}
