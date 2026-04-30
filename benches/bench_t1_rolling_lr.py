"""Tier-1 rolling LR bench: production path (post-promotion of T1.4)."""

from __future__ import annotations

import pytest
import polars_ds as pds


@pytest.mark.benchmark(group="t1_4_rolling_lr")
def test_rolling_lr_prod(benchmark, rolling_lr_df):
    df = rolling_lr_df

    @benchmark
    def run():
        return df.select(
            pds.rolling_lin_reg(
                "x1",
                "x2",
                "x3",
                target="y",
                add_bias=False,
                window_size=30,
                min_valid_rows=10,
                null_policy="skip",
            ).alias("coeffs")
        )
