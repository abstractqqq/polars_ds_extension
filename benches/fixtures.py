"""
Seeded data generators for Tier-1 polars-ds benchmarks.

All generators use numpy.random.default_rng(42) and convert directly to polars,
avoiding intermediate pandas allocations.

Memory notes
------------
- make_glm_irls_df():   ~240 MB  (3 M rows, 7 cols; GLM_SCALE_DOWN=True by default)
- make_knn_radius_df(): ~48 MB   (1 M rows, 6 cols f64 + uint32)
- make_entropy_series(): ~8 MB   (1 M f64 values)
- make_rolling_lr_df(): ~28 MB   (500 k rows, 5 cols f64)

If GLM_SCALE_DOWN is set to False (env var GLM_FULL=1), make_glm_irls_df() generates
30 M rows (≈2 GB).  All four fixtures together consume ~4 GB in that mode.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl

SEED = 42

# Honour GLM_FULL=1 env var to opt into the full 30 M-row fixture.
_GLM_FULL: bool = os.getenv("GLM_FULL", "0") == "1"

# Default (scale-down): 100 k groups × 30 rows = 3 M rows, ~240 MB.
# Full:                  1 M groups × 30 rows = 30 M rows, ~2 GB.
_GLM_N_GROUPS: int = 1_000_000 if _GLM_FULL else 100_000
_GLM_ROWS_PER_GROUP: int = 30
_GLM_TOTAL_ROWS: int = _GLM_N_GROUPS * _GLM_ROWS_PER_GROUP


def _rng() -> np.random.Generator:
    """Return a fresh Generator seeded at SEED=42."""
    return np.random.default_rng(SEED)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Fixture 1 — GLM IRLS group_by workload
# ---------------------------------------------------------------------------


def make_glm_irls_df() -> pl.DataFrame:
    """
    group_by GLM (IRLS, Binomial) workload.

    Rows: {_GLM_TOTAL_ROWS}
    Groups: {_GLM_N_GROUPS} (int32, repeated {_GLM_ROWS_PER_GROUP} times each).
    Features: x1..x5 (f64, standard normal).
    Target: y (int32, 0/1) sampled from P(y=1)=sigmoid(0.5*x1 - 0.3*x2 + 0.1*x3).

    Scale-down flag: set env GLM_FULL=1 for 30 M rows.  Default is 3 M rows.
    """
    rng = _rng()
    n = _GLM_TOTAL_ROWS

    group = np.repeat(np.arange(_GLM_N_GROUPS, dtype=np.int32), _GLM_ROWS_PER_GROUP)

    x = rng.standard_normal((n, 5))
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

    logit = 0.5 * x1 - 0.3 * x2 + 0.1 * x3
    prob = _sigmoid(logit)
    y = rng.binomial(1, prob).astype(np.int32)

    return pl.DataFrame(
        {
            "group": pl.Series("group", group, dtype=pl.Int32),
            "x1": pl.Series("x1", x[:, 0], dtype=pl.Float64),
            "x2": pl.Series("x2", x[:, 1], dtype=pl.Float64),
            "x3": pl.Series("x3", x[:, 2], dtype=pl.Float64),
            "x4": pl.Series("x4", x[:, 3], dtype=pl.Float64),
            "x5": pl.Series("x5", x[:, 4], dtype=pl.Float64),
            "y": pl.Series("y", y, dtype=pl.Int32),
        }
    )


make_glm_irls_df.__doc__ = make_glm_irls_df.__doc__.format(  # type: ignore[union-attr]
    _GLM_TOTAL_ROWS=_GLM_TOTAL_ROWS,
    _GLM_N_GROUPS=_GLM_N_GROUPS,
    _GLM_ROWS_PER_GROUP=_GLM_ROWS_PER_GROUP,
)


# ---------------------------------------------------------------------------
# Fixture 2 — KNN radius query workload
# ---------------------------------------------------------------------------


def make_knn_radius_df() -> pl.DataFrame:
    """
    KNN radius query workload.

    Rows: 1,000,000.
    Columns: idx (uint32, 0..999999), x1..x5 (f64, uniform [0,1)).
    Intended use: radius search with r=0.3 squared-L2.
    """
    rng = _rng()
    n = 1_000_000

    idx = np.arange(n, dtype=np.uint32)
    coords = rng.random((n, 5))

    return pl.DataFrame(
        {
            "idx": pl.Series("idx", idx, dtype=pl.UInt32),
            "x1": pl.Series("x1", coords[:, 0], dtype=pl.Float64),
            "x2": pl.Series("x2", coords[:, 1], dtype=pl.Float64),
            "x3": pl.Series("x3", coords[:, 2], dtype=pl.Float64),
            "x4": pl.Series("x4", coords[:, 3], dtype=pl.Float64),
            "x5": pl.Series("x5", coords[:, 4], dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Fixture 3 — Entropy / ts-features single-series workload
# ---------------------------------------------------------------------------


def make_entropy_series() -> pl.Series:
    """
    Single 1,000,000-row standard-normal f64 Series named "ts".

    Used for approximate entropy, sample entropy, and rolling ts-feature benchmarks.
    """
    rng = _rng()
    data = rng.standard_normal(1_000_000)
    return pl.Series("ts", data, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Fixture 4 — Rolling LR with leading-null prefix
# ---------------------------------------------------------------------------


def make_rolling_lr_df() -> pl.DataFrame:
    """
    Rolling LR workload with a leading-null prefix.

    Total rows: 500,000.
      - First 100,000 rows: x1=x2=x3=null, y=non-null (intercept-only region).
      - Remaining 400,000: x1,x2,x3 standard normal; y = 0.5*x1 - 0.3*x2 + 0.2*x3 + noise.

    All rows have non-null y.
    """
    rng = _rng()
    n_null = 100_000
    n_data = 400_000

    x = rng.standard_normal((n_data, 3))
    noise = 0.01 * rng.standard_normal(n_data)
    y_data = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2] + noise

    # Null prefix for features; y is always present.
    y_null = rng.standard_normal(n_null)
    y_full = np.concatenate([y_null, y_data])

    x1_full = np.concatenate([np.full(n_null, np.nan), x[:, 0]])
    x2_full = np.concatenate([np.full(n_null, np.nan), x[:, 1]])
    x3_full = np.concatenate([np.full(n_null, np.nan), x[:, 2]])

    # polars stores NaN as NaN (not null) for Float64; use nan_to_null() to
    # produce proper Polars nulls in the leading prefix region.
    def _with_nulls(name: str, arr: np.ndarray) -> pl.Series:
        return pl.Series(name, arr, dtype=pl.Float64).fill_nan(None)

    return pl.DataFrame(
        {
            "x1": _with_nulls("x1", x1_full),
            "x2": _with_nulls("x2", x2_full),
            "x3": _with_nulls("x3", x3_full),
            "y": pl.Series("y", y_full, dtype=pl.Float64),
        }
    )
