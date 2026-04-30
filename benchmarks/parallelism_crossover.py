"""
Rayon-vs-sequential crossover benchmark for polars_ds.

Purpose
-------
Find the input sizes at which parallel (rayon) execution becomes faster than
sequential execution for two hot paths:

  1. series_to_slice proxy  — column-copy overhead dominates at small sizes;
     proxied by a flat (no group_by) pds.lin_reg() call with add_bias=True.
  2. predict matmul proxy   — faer matrix-multiply; proxied by LR.predict() on
     a pre-built numpy array.

This file documents BASELINE behaviour (build as-is, thresholds hard-coded to
4096 by default).  To find the empirical crossover:

  Step 1 — force always-parallel:
      PDS_SMALL_INPUT_THRESHOLD=0 PDS_PARALLEL_MATMUL_THRESHOLD=0 \\
          maturin develop --release && \\
          python benchmarks/parallelism_crossover.py > parallel.txt

  Step 2 — force always-sequential:
      PDS_SMALL_INPUT_THRESHOLD=999999999 PDS_PARALLEL_MATMUL_THRESHOLD=999999999 \\
          maturin develop --release && \\
          python benchmarks/parallelism_crossover.py > sequential.txt

  Step 3 — diff the two tables.  The crossover is the size at which
  sequential first becomes slower than parallel (i.e. parallel wins above
  that size).  Set SMALL_INPUT_THRESHOLD and PARALLEL_MATMUL_THRESHOLD in
  src/utils/parallelism.rs to that size.

Caveat
------
Results depend heavily on hardware (core count, cache topology, NUMA).
CI uses ubuntu-latest (typically 2 cores on GitHub Actions shared runners).
Run on the target deployment hardware before finalising the constants.

No third-party dependencies beyond polars, numpy, polars_ds.
"""

from __future__ import annotations

import statistics
import time
from typing import List, Tuple

import numpy as np
import polars as pl
import polars_ds as pds
from polars_ds.linear_models import LR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLS = 4
SIZES = [64, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576]

# iterations per measurement — large enough for median to converge, small
# enough that the largest size finishes quickly.
ITERS_LIN_REG = 200   # series_to_slice proxy
ITERS_PREDICT = 500   # predict matmul proxy (lighter per-call)

WARMUP = 10  # throw-away iterations before timing starts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: int, cols: int) -> pl.DataFrame:
    """Build a Polars DataFrame with `cols` float feature columns + 1 target."""
    feature_exprs = [pds.random(0.0, 1.0).alias(f"x{i}") for i in range(cols)]
    df = pds.frame(size=rows).select(*feature_exprs)
    # target = sum of features + tiny noise
    y = df.select(
        (pl.sum_horizontal([pl.col(f"x{i}") for i in range(cols)]) + pds.random() * 1e-4).alias("y")
    ).get_column("y")
    return df.with_columns(y)


def _feature_cols(cols: int) -> List[str]:
    return [f"x{i}" for i in range(cols)]


def _measure_ns(fn, iters: int) -> List[float]:
    """Return a list of per-call times in nanoseconds."""
    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return times


def _median_us(times_ns: List[float]) -> float:
    return statistics.median(times_ns) / 1_000.0


# ---------------------------------------------------------------------------
# Benchmark 1 — series_to_slice proxy (flat lin_reg)
# ---------------------------------------------------------------------------

def bench_series_to_slice(rows: int, cols: int) -> Tuple[float, int]:
    """
    Proxy for the series_to_slice hot path.

    pds.lin_reg with add_bias=True on a flat DataFrame exercises the
    column-copy path (series_to_slice) on every call.  At small input sizes
    this per-call overhead dominates the actual solve time.

    Returns (median_us_per_call, iters).
    """
    df = _make_df(rows, cols)
    feats = _feature_cols(cols)

    def _call():
        df.select(pds.lin_reg(*feats, target="y", add_bias=True))

    # warmup
    _measure_ns(_call, WARMUP)
    times = _measure_ns(_call, ITERS_LIN_REG)
    return _median_us(times), ITERS_LIN_REG


# ---------------------------------------------------------------------------
# Benchmark 2 — predict matmul proxy
# ---------------------------------------------------------------------------

def bench_predict(rows: int, cols: int) -> Tuple[float, int]:
    """
    Proxy for the predict / faer-matmul hot path.

    Fit LR once, then call .predict() repeatedly on a pre-built C-contiguous
    float64 numpy array.  This bypasses the series_to_slice path and focuses
    purely on the matmul dispatch (Par::rayon vs Par::Seq in faer).

    Returns (median_us_per_call, iters).
    """
    df = _make_df(rows, cols)
    feats = _feature_cols(cols)
    X = df.select(feats).to_numpy(order="c")
    y = df.get_column("y").to_numpy()

    model = LR(has_bias=True)
    model.fit(X, y)

    def _call():
        model.predict(X)

    # warmup
    _measure_ns(_call, WARMUP)
    times = _measure_ns(_call, ITERS_PREDICT)
    return _median_us(times), ITERS_PREDICT


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _md_table(
    header: List[str],
    rows: List[List[str]],
) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import platform
    import os

    print(f"polars_ds parallelism crossover benchmark")
    print(f"Python platform : {platform.platform()}")
    print(f"CPU cores (logical): {os.cpu_count()}")
    print(f"Fixed cols per frame : {COLS}")
    print()

    header_slc = ["rows", "total_cells", "median_us (lin_reg)", "iters"]
    header_mat = ["rows", "total_cells", "median_us (predict)", "iters"]
    rows_slc: List[List[str]] = []
    rows_mat: List[List[str]] = []

    for size in SIZES:
        total = size * COLS
        print(f"  rows={size:>10,}  total_cells={total:>12,} ...", end="", flush=True)

        us_slc, it_slc = bench_series_to_slice(size, COLS)
        us_mat, it_mat = bench_predict(size, COLS)

        rows_slc.append([f"{size:,}", f"{total:,}", f"{us_slc:,.2f}", str(it_slc)])
        rows_mat.append([f"{size:,}", f"{total:,}", f"{us_mat:,.2f}", str(it_mat)])

        print(f"  lin_reg={us_slc:.2f}µs  predict={us_mat:.2f}µs")

    print()
    print("### series_to_slice proxy (pds.lin_reg, add_bias=True, flat frame)")
    print()
    print(_md_table(header_slc, rows_slc))
    print()
    print("### predict matmul proxy (LR.predict on numpy array)")
    print()
    print(_md_table(header_mat, rows_mat))
    print()
    print(
        "To find the crossover: re-run with PDS_SMALL_INPUT_THRESHOLD=0 (parallel-forced) "
        "and PDS_SMALL_INPUT_THRESHOLD=999999999 (sequential-forced) after rebuilding, "
        "then compare the two tables to identify where sequential first becomes slower."
    )
