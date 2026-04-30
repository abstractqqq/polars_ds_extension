"""
Parity case for pl_query_radius_ptwise.
Bug: serial-path ListPrimitiveChunkedBuilder allocated with data.len() (= nrows*ncols)
     instead of nrows.  Fixed in pl_query_radius_ptwise_new_expr.
"""
from __future__ import annotations
import polars as pl
from tests.parity_oracle import register


@register("pl_query_radius_ptwise", fixtures=["tiny_clean", "medium_multichunk"])
def _case(df: pl.DataFrame):
    """
    Mirror what python/polars_ds/exprs/expr_knn.py:query_radius_ptwise builds.
    Signature: pl_query_radius_ptwise(inputs=[idx_u32, *features], kwargs={r, metric, parallel, sort})

    tiny_clean cols: x, y, weight, label, cat   → features = x, y
    medium_multichunk cols: x1, x2, x3, label, cat → features = x1, x2, x3
    """
    # Detect which fixture we have by available columns
    if "x" in df.columns and "y" in df.columns and "x1" not in df.columns:
        # tiny_clean: 2-D, synthesise uint32 idx
        idx = pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("_idx")
        args = [idx.cast(pl.UInt32), pl.col("x").cast(pl.Float64), pl.col("y").cast(pl.Float64)]
        radius = 1.5
    else:
        # medium_multichunk: 3-D
        idx = pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("_idx")
        args = [
            idx.cast(pl.UInt32),
            pl.col("x1").cast(pl.Float64),
            pl.col("x2").cast(pl.Float64),
            pl.col("x3").cast(pl.Float64),
        ]
        radius = 1.0

    kwargs = {"r": radius, "metric": "sql2", "parallel": False, "sort": True}
    extra = {"is_elementwise": False, "returns_scalar": False}
    return args, kwargs, extra
