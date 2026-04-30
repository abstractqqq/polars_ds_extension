"""
Parity case: pl_approximate_entropy (par_bridge old) vs pl_approximate_entropy_new_expr (split_offsets new).

Both `parallel=True` so the hot path is exercised.
"""

from __future__ import annotations

import polars as pl

from tests.parity_oracle import register


def _build_args(df: pl.DataFrame):
    """Build plugin args/kwargs for pl_approximate_entropy / _new_expr.

    Mirrors query_approx_entropy(ts, m=2, filtering_level=0.2, scale_by_std=True, parallel=True).
    """
    ts = pl.col("x").cast(pl.Float64)
    m = 2
    filtering_level = 0.2

    t = df.lazy().select(ts).collect()["x"]
    std_val = t.std(ddof=0)
    r_lit = pl.lit(filtering_level * std_val, dtype=pl.Float64)

    rows = len(t) - m + 1
    data = [r_lit, ts.slice(0, length=rows).cast(pl.Float64)]
    data.extend(
        ts.shift(-i).slice(0, length=rows).cast(pl.Float64).alias(str(i))
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


@register("pl_approximate_entropy", fixtures=["tiny_clean", "medium_multichunk"])
def _case(df: pl.DataFrame):
    return _build_args(df)
