use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rapidfuzz::distance::{hamming, levenshtein};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct NearestStrKwargs {
    pub(crate) word: String,
    pub(crate) metric: String,
    pub(crate) threshold: usize,
}

fn levenshtein_nearest<'a>(s: &'a StringChunked, cutoff: usize, word: String) -> Option<&'a str> {
    let batched = levenshtein::BatchComparator::new(word.chars());
    // Most similar == having smallest distance
    let mut best: usize = usize::MAX;
    let mut actual_cutoff = levenshtein::Args::default().score_cutoff(cutoff);
    let mut nearest_str: Option<&str> = None;
    for arr in s.downcast_iter() {
        for w in arr.values_iter() {
            if let Some(d) = batched.distance_with_args(w.chars(), &actual_cutoff) {
                if d == 0 {
                    return Some(w);
                } else if d < best {
                    best = d;
                    nearest_str = Some(w);
                    actual_cutoff = actual_cutoff.score_cutoff(best);
                }
            }
        }
    }
    nearest_str
}

fn hamming_nearest<'a>(s: &'a StringChunked, cutoff: usize, word: String) -> Option<&'a str> {
    let batched = hamming::BatchComparator::new(word.chars());
    let mut actual_cutoff = hamming::Args::default().score_cutoff(cutoff);
    let mut best: usize = usize::MAX;
    let mut nearest_str: Option<&str> = None;

    for arr in s.downcast_iter() {
        for w in arr.values_iter() {
            if let Ok(ss) = batched.distance_with_args(w.chars(), &actual_cutoff) {
                if let Some(d) = ss {
                    if d == 0 {
                        return Some(w);
                    } else if d < best {
                        best = d;
                        nearest_str = Some(w);
                        actual_cutoff = actual_cutoff.score_cutoff(best);
                    }
                }
            }
        }
    }
    nearest_str
}

#[polars_expr(output_type=String)]
pub fn pl_nearest_str(inputs: &[Series], kwargs: NearestStrKwargs) -> PolarsResult<Series> {
    let s = inputs[0].str()?;
    let word = kwargs.word;
    let cutoff = kwargs.threshold;
    let func = match kwargs.metric.as_str() {
        "hamming" => hamming_nearest,
        _ => levenshtein_nearest,
    };
    let mut builder = StringChunkedBuilder::new(s.name().clone(), 1);
    builder.append_option(func(s, cutoff, word));
    let ca = builder.finish();
    Ok(ca.into_series())
}
