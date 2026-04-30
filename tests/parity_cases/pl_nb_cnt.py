"""
Parity case for pl_nb_cnt.
Bug: parallel branch allocates PrimitiveChunkedBuilder with `nrows` (total rows)
     instead of `len` (per-thread share).  Fixed in pl_nb_cnt_new_expr.
"""
from __future__ import annotations
import polars as pl
from tests.parity_oracle import register


@register("pl_nb_cnt", fixtures=["tiny_clean", "medium_multichunk"])
def _case(df: pl.DataFrame):
    """
    Mirror what python/polars_ds/exprs/expr_knn.py:query_nb_cnt builds.
    Signature: pl_nb_cnt(inputs=[radius_f64, *features], kwargs={k, metric, parallel, skip_eval, skip_data})

    tiny_clean cols: x, y, weight, label, cat   → features = x, y
    medium_multichunk cols: x1, x2, x3, label, cat → features = x1, x2, x3
    """
    if "x" in df.columns and "y" in df.columns and "x1" not in df.columns:
        # tiny_clean: 2-D
        radius = 1.5
        rad_expr = pl.lit(pl.Series(values=[radius], dtype=pl.Float64))
        args = [rad_expr, pl.col("x").cast(pl.Float64), pl.col("y").cast(pl.Float64)]
    else:
        # medium_multichunk: 3-D
        radius = 1.0
        rad_expr = pl.lit(pl.Series(values=[radius], dtype=pl.Float64))
        args = [
            rad_expr,
            pl.col("x1").cast(pl.Float64),
            pl.col("x2").cast(pl.Float64),
            pl.col("x3").cast(pl.Float64),
        ]

    kwargs = {"k": 0, "metric": "sql2", "parallel": False, "skip_eval": False, "skip_data": False}
    extra = {"is_elementwise": False, "returns_scalar": False}
    return args, kwargs, extra
