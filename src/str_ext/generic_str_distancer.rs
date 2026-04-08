use polars::{
    prelude::{
        arity::{binary_elementwise_values, unary_elementwise_values}, DataType, Float64Chunked, Series, StringChunked,
        UInt32Chunked,
    },
    series::IntoSeries,
};
use pyo3_polars::export::polars_core::{
    utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
    POOL,
};
/// Polars Series-wise generic str distancers
use rapidfuzz::distance::{damerau_levenshtein, jaro, lcs_seq, levenshtein, osa};

use crate::utils::split_offsets;

// Str Distance Related Helper Functions
pub trait StdBatchedStrDistancer {
    fn distance(&self, s: &str) -> u32;
    fn normalized_similarity(&self, s: &str) -> f64;
}

macro_rules! StdBatchedStrDistanceImpl {
    ($batch_struct: ty) => {
        impl StdBatchedStrDistancer for $batch_struct {
            fn distance(&self, s: &str) -> u32 {
                self.distance(s.chars()) as u32
            }

            fn normalized_similarity(&self, s: &str) -> f64 {
                self.normalized_similarity(s.chars())
            }
        }
    };
}

macro_rules! StdBatchedBytesDistanceImpl {
    ($batch_struct: ty) => {
        impl StdBatchedStrDistancer for $batch_struct {
            fn distance(&self, s: &str) -> u32 {
                self.distance(s.as_bytes().iter().copied()) as u32
            }

            fn normalized_similarity(&self, s: &str) -> f64 {
                self.normalized_similarity(s.as_bytes().iter().copied())
            }
        }
    };
}

StdBatchedStrDistanceImpl!(lcs_seq::BatchComparator<char>);
StdBatchedStrDistanceImpl!(osa::BatchComparator<char>);
StdBatchedStrDistanceImpl!(levenshtein::BatchComparator<char>);
StdBatchedStrDistanceImpl!(damerau_levenshtein::BatchComparator<char>);
StdBatchedStrDistanceImpl!(jaro::BatchComparator<char>);

StdBatchedBytesDistanceImpl!(levenshtein::BatchComparator<u8>);
StdBatchedBytesDistanceImpl!(damerau_levenshtein::BatchComparator<u8>);
// -------------------------------------------------------------------------------------

pub fn generic_batched_distance<T>(batched: T, ca: &StringChunked, parallel: bool) -> Series
where
    T: StdBatchedStrDistancer + std::marker::Sync,
{
    let out: UInt32Chunked = if parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(ca.len(), n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let s1 = ca.slice(offset as i64, len);
            let out: UInt32Chunked =
                unary_elementwise_values(&s1, |s| batched.distance(s));
            out.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        UInt32Chunked::from_chunk_iter(ca.name().clone(), chunks.into_iter().flatten())
    } else {
        unary_elementwise_values(ca, |s| batched.distance(s))
    };
    out.into_series()
}

pub fn generic_batched_sim<T>(batched: T, ca: &StringChunked, parallel: bool) -> Series
where
    T: StdBatchedStrDistancer + std::marker::Sync,
{
    let out: Float64Chunked = if parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(ca.len(), n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let s1 = ca.slice(offset as i64, len);
            let out: Float64Chunked = unary_elementwise_values(&s1, |s| batched.normalized_similarity(s));
            out.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        Float64Chunked::from_chunk_iter(ca.name().clone(), chunks.into_iter().flatten())
    } else {
        unary_elementwise_values(ca, |s| batched.normalized_similarity(s))
    };
    out.into_series()
}

pub fn generic_binary_distance<F>(
    func: F,
    ca1: &StringChunked,
    ca2: &StringChunked,
    parallel: bool,
) -> Series 
where
    F: Fn(&str, &str) -> u32 + Copy + Send + Sync,
{
    let out: UInt32Chunked = if parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(ca1.len(), n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let s1 = ca1.slice(offset as i64, len);
            let s2 = ca2.slice(offset as i64, len);
            let out: UInt32Chunked = binary_elementwise_values(&s1, &s2, func);
            out.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        UInt32Chunked::from_chunk_iter(ca1.name().clone(), chunks.into_iter().flatten())
    } else {
        binary_elementwise_values(ca1, ca2, func)
    };
    out.into_series()
}

pub fn generic_binary_sim<F>(
    func: F,
    ca1: &StringChunked,
    ca2: &StringChunked,
    parallel: bool,
) -> Series 
where
    F: Fn(&str, &str) -> f64 + Copy + Send + Sync,
{
    let out: Float64Chunked = if parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(ca1.len(), n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let s1 = ca1.slice(offset as i64, len);
            let s2 = ca2.slice(offset as i64, len);
            let out: Float64Chunked = binary_elementwise_values(&s1, &s2, func);
            out.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        Float64Chunked::from_chunk_iter(ca1.name().clone(), chunks.into_iter().flatten())
    } else {
        binary_elementwise_values(ca1, ca2, func)
    };
    out.into_series()
}
