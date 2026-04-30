"""Tier-1 entropies bench: T1.3 par_bridge → split_offsets."""
from __future__ import annotations

import polars as pl
import pytest

from polars_ds._utils import pl_plugin


def _build_args(ts: pl.Series, *, m: int = 2, ratio: float = 0.2):
    std_val = ts.std(ddof=0)
    r_val = ratio * std_val
    rows = len(ts) - m + 1

    df = ts.to_frame()
    name = ts.name
    base = pl.col(name).slice(0, length=rows)
    shifts = [pl.col(name).shift(-i).slice(0, length=rows).alias(str(i)) for i in range(1, m + 1)]
    data = [pl.lit(r_val, dtype=pl.Float64), base, *shifts]
    kwargs = {"k": 0, "metric": "inf", "parallel": True}
    extra = {"returns_scalar": True, "pass_name_to_apply": True}
    return df, data, kwargs, extra


@pytest.mark.benchmark(group="t1_3_approximate_entropy")
def test_approx_entropy_old(benchmark, entropy_series):
    df, args, kwargs, extra = _build_args(entropy_series)

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_approximate_entropy",
                                   args=args, kwargs=kwargs, **extra))


@pytest.mark.benchmark(group="t1_3_approximate_entropy")
def test_approx_entropy_new(benchmark, entropy_series):
    df, args, kwargs, extra = _build_args(entropy_series)

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_approximate_entropy_new_expr",
                                   args=args, kwargs=kwargs, **extra))


@pytest.mark.benchmark(group="t1_3_sample_entropy")
def test_sample_entropy_old(benchmark, entropy_series):
    df, args, kwargs, extra = _build_args(entropy_series)

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_sample_entropy",
                                   args=args, kwargs=kwargs, **extra))


@pytest.mark.benchmark(group="t1_3_sample_entropy")
def test_sample_entropy_new(benchmark, entropy_series):
    df, args, kwargs, extra = _build_args(entropy_series)

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_sample_entropy_new_expr",
                                   args=args, kwargs=kwargs, **extra))
