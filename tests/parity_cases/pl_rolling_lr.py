"""
Parity case: pl_rolling_lr (faer_rolling_skipping_lr_old) vs
             pl_rolling_lr_new_expr (faer_rolling_skipping_lr_new).

Uses null_policy="skip" to exercise the skipping-LR code path
(both the O(n*leading_nulls) startup scan and the incremental counter).

Fixtures:
  tiny_with_nulls   — 20 rows, ~20% nulls; exercises the startup-scan hotspot.
  medium_with_nulls — 2000 rows, ~5% nulls; exercises steady-state Woodbury updates.

The parity oracle calls the build_call for every fixture in the list using the
same function.  Because tiny_with_nulls and medium_with_nulls have different
column schemas, we dispatch inside build_call based on what columns are present.
"""

from __future__ import annotations

import polars as pl

from tests.parity_oracle import register


def _build_call(df: pl.DataFrame):
    """Dispatch on available columns."""
    if "x" in df.columns and "y" in df.columns:
        # tiny_with_nulls schema: x (Float64 w/ nulls), y (Float64 w/ nulls)
        y = pl.col("y").cast(pl.Float64)
        x = pl.col("x").cast(pl.Float64)
        args = [y, x]
        kwargs = {
            "null_policy": "skip",
            "n": 5,
            "bias": False,
            "lambda": 0.0,
            "min_size": 3,
        }
    else:
        # medium_with_nulls schema: x1 (Float64 w/ nulls), x2 (Float64), label (Int32)
        y = pl.col("label").cast(pl.Float64)
        x1 = pl.col("x1").cast(pl.Float64)
        x2 = pl.col("x2").cast(pl.Float64)
        args = [y, x1, x2]
        kwargs = {
            "null_policy": "skip",
            "n": 10,
            "bias": False,
            "lambda": 0.0,
            "min_size": 5,
        }
    extra = {"pass_name_to_apply": True}
    return args, kwargs, extra


@register("pl_rolling_lr", fixtures=["tiny_with_nulls", "medium_with_nulls"])
def _case(df: pl.DataFrame):
    return _build_call(df)
