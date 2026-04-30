"""
Parity case: pl_sample_entropy (par_bridge old) vs pl_sample_entropy_new_expr (split_offsets new).

Both `parallel=True` so the hot path is exercised.
"""

from __future__ import annotations

import polars as pl

from tests.parity_oracle import register


def _build_args(df: pl.DataFrame):
    """Build plugin args/kwargs for pl_sample_entropy / _new_expr.

    Mirrors query_sample_entropy(ts, ratio=0.2, m=2, parallel=True).
    Pick first f64 column as the time series so this works across fixtures.
    """
    f64_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Float64]
    col_name = f64_cols[0] if f64_cols else df.columns[0]
    ts = pl.col(col_name).cast(pl.Float64)
    ratio = 0.2
    m = 2

    t = df.lazy().select(ts).collect()[col_name]
    std_val = t.std(ddof=0)
    r_lit = pl.lit(ratio * std_val, dtype=pl.Float64)

    rows = len(t) - m + 1
    data = [r_lit, ts.slice(0, length=rows)]
    data.extend(
        ts.shift(-i).slice(0, length=rows).alias(str(i))
        for i in range(1, m + 1)
    )

    kwargs = {
        "k": 0,
        "metric": "inf",
        "parallel": True,
    }
    extra = {
        "returns_scalar": True,
        "pass_name_to_apply": True,
    }
    return data, kwargs, extra


@register("pl_sample_entropy", fixtures=["tiny_clean", "medium_multichunk"])
def _case(df: pl.DataFrame):
    return _build_args(df)
