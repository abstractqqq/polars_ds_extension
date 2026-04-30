//! Parallelism thresholds.
//!
//! Below these total-element counts, hot paths run sequentially to avoid
//! rayon spawn cost dominating the actual work. Calibrated empirically; see
//! `benchmarks/parallelism_crossover.py`. Crossover is the input size at
//! which parallel matches sequential wall-clock; values below are placeholders
//! pending the benchmark run on CI hardware (ubuntu-latest).
//!
//! Compile-time overrides via env vars:
//!   PDS_SMALL_INPUT_THRESHOLD       — int, total cells for series_to_slice gate
//!   PDS_PARALLEL_MATMUL_THRESHOLD   — int, total cells for predict matmul gate

const fn parse_usize_or(s: Option<&str>, default: usize) -> usize {
    match s {
        None => default,
        Some(s) => {
            let bytes = s.as_bytes();
            let mut i = 0usize;
            let mut acc: usize = 0;
            while i < bytes.len() {
                let b = bytes[i];
                assert!(
                    b >= b'0' && b <= b'9',
                    "env var must be base-10 unsigned integer"
                );
                acc = acc * 10 + (b - b'0') as usize;
                i += 1;
            }
            acc
        }
    }
}

/// Below this total cell count (rows * cols) `series_to_slice` runs
/// sequentially instead of dispatching rayon over columns.
pub(crate) const SMALL_INPUT_THRESHOLD: usize =
    parse_usize_or(option_env!("PDS_SMALL_INPUT_THRESHOLD"), 4096);

/// Below this total cell count (rows * cols) faer matmul in `predict`
/// uses `Par::Seq` instead of `Par::rayon(0)`.
pub(crate) const PARALLEL_MATMUL_THRESHOLD: usize =
    parse_usize_or(option_env!("PDS_PARALLEL_MATMUL_THRESHOLD"), 4096);
