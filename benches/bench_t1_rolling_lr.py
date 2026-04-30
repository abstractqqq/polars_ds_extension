"""Tier-1 rolling LR bench: T1.4 startup-scan O(n*leading_nulls) → O(n)."""
from __future__ import annotations

import polars as pl
import pytest
import polars_ds as pds


@pytest.mark.benchmark(group="t1_4_rolling_lr")
def test_rolling_lr_old(benchmark, rolling_lr_df):
    df = rolling_lr_df

    @benchmark
    def run():
        return df.select(
            pds.rolling_lin_reg(
                "x1", "x2", "x3",
                target="y",
                add_bias=False,
                window_size=30,
                min_valid_rows=10,
                null_policy="skip",
            ).alias("coeffs")
        )


@pytest.mark.benchmark(group="t1_4_rolling_lr")
def test_rolling_lr_new(benchmark, rolling_lr_df):
    """Bench the _new variant by calling pl_rolling_lr_new_expr directly."""
    from polars_ds._utils import pl_plugin

    df = rolling_lr_df
    args = [
        pl.col("y").cast(pl.Float64),
        pl.col("x1").cast(pl.Float64),
        pl.col("x2").cast(pl.Float64),
        pl.col("x3").cast(pl.Float64),
    ]
    kwargs = {
        "null_policy": "skip",
        "solver": "qr",
        "lambda_": 0.0,
        "min_size": 10,
        "n": 30,
    }

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_rolling_lr_new_expr",
                                   args=args, kwargs=kwargs,
                                   changes_length=False).alias("coeffs"))
