"""
Parity case for pl_query_radius_ptwise_null_safe (bug-029 null-guard fix).

Compares:
  old: pl_query_radius_ptwise_null_safe      — accepts [idx, keep_mask, *feats],
                                               ignores keep_mask, uses old non-null-safe logic
                                               (with capacity fix applied)
  new: pl_query_radius_ptwise_null_safe_new_expr — same layout, adds null-guard via keep_mask

On null-free fixtures (tiny_clean, medium_multichunk) the two paths must produce
bit-identical output, confirming the null-guard does not alter behavior when there
are no nulls.

For null-safety regression (inputs WITH nulls), see:
  tests/test_many.py::test_radius_ptwise_with_nulls_no_panic
"""
from __future__ import annotations
import polars as pl
from tests.parity_oracle import register


@register("pl_query_radius_ptwise_null_safe", fixtures=["tiny_clean", "medium_multichunk"])
def _case(df: pl.DataFrame):
    """
    Input layout for both old and new:
      inputs[0]  : idx (u32)
      inputs[1]  : keep_mask (bool, all-True on null-free fixtures)
      inputs[2:] : feature columns (f64)

    tiny_clean cols     : x, y, weight, label, cat  -> features = x, y
    medium_multichunk   : x1, x2, x3, label, cat   -> features = x1, x2, x3
    """
    if "x" in df.columns and "y" in df.columns and "x1" not in df.columns:
        # tiny_clean: 2-D
        idx = pl.int_range(0, pl.len(), dtype=pl.UInt32)
        feat_exprs = [pl.col("x").cast(pl.Float64), pl.col("y").cast(pl.Float64)]
        radius = 1.5
    else:
        # medium_multichunk: 3-D
        idx = pl.int_range(0, pl.len(), dtype=pl.UInt32)
        feat_exprs = [
            pl.col("x1").cast(pl.Float64),
            pl.col("x2").cast(pl.Float64),
            pl.col("x3").cast(pl.Float64),
        ]
        radius = 1.0

    # On null-free data, keep_mask is all-True
    keep_mask = pl.all_horizontal(f.is_not_null() for f in feat_exprs)

    args = [idx, keep_mask, *feat_exprs]
    kwargs = {"r": radius, "metric": "sql2", "parallel": False, "sort": True}
    extra = {"is_elementwise": False, "returns_scalar": False}
    return args, kwargs, extra
