"""Tier-1 KNN bench: T1.2 builder capacities + B4 null-safe."""
from __future__ import annotations

import polars as pl
import pytest

from polars_ds._utils import pl_plugin


@pytest.mark.benchmark(group="t1_2_knn_radius_ptwise_serial")
def test_radius_ptwise_old_serial(benchmark, knn_radius_df):
    df = knn_radius_df
    feats = [pl.col(f"x{i}") for i in range(1, 6)]
    args_old = [pl.col("idx").cast(pl.UInt32), *feats]
    kwargs = {"r": 0.3, "metric": "sql2", "parallel": False, "sort": True}

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_query_radius_ptwise",
                                   args=args_old, kwargs=kwargs,
                                   changes_length=False))


@pytest.mark.benchmark(group="t1_2_knn_radius_ptwise_serial")
def test_radius_ptwise_new_serial(benchmark, knn_radius_df):
    df = knn_radius_df
    feats = [pl.col(f"x{i}") for i in range(1, 6)]
    args_new = [pl.col("idx").cast(pl.UInt32), *feats]
    kwargs = {"r": 0.3, "metric": "sql2", "parallel": False, "sort": True}

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_query_radius_ptwise_new_expr",
                                   args=args_new, kwargs=kwargs,
                                   changes_length=False))


@pytest.mark.benchmark(group="t1_2_knn_nb_cnt_parallel")
def test_nb_cnt_old_parallel(benchmark, knn_radius_df):
    df = knn_radius_df
    feats = [pl.col(f"x{i}") for i in range(1, 6)]
    args = [pl.lit(0.3, dtype=pl.Float64), *feats]
    kwargs = {"k": 0, "metric": "sql2", "parallel": True}

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_nb_cnt", args=args,
                                   kwargs=kwargs, changes_length=False))


@pytest.mark.benchmark(group="t1_2_knn_nb_cnt_parallel")
def test_nb_cnt_new_parallel(benchmark, knn_radius_df):
    df = knn_radius_df
    feats = [pl.col(f"x{i}") for i in range(1, 6)]
    args = [pl.lit(0.3, dtype=pl.Float64), *feats]
    kwargs = {"k": 0, "metric": "sql2", "parallel": True}

    @benchmark
    def run():
        return df.select(pl_plugin(symbol="pl_nb_cnt_new_expr", args=args,
                                   kwargs=kwargs, changes_length=False))
