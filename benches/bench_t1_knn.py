"""Tier-1 KNN bench: production path (post-promotion)."""

from __future__ import annotations

import pytest
import polars_ds as pds


_R = 0.01  # tuned: ~25 expected neighbors per query in 5D uniform [0,1)^5


@pytest.mark.benchmark(group="t1_2_knn_radius_ptwise")
def test_radius_ptwise_prod(benchmark, knn_radius_df):
    df = knn_radius_df

    @benchmark
    def run():
        return df.select(
            pds.query_radius_ptwise(
                "x1",
                "x2",
                "x3",
                "x4",
                "x5",
                index="idx",
                r=_R,
                dist="sql2",
                parallel=False,
                sort=True,
            ).alias("nbrs")
        )


@pytest.mark.benchmark(group="t1_2_knn_nb_cnt")
def test_nb_cnt_prod(benchmark, knn_radius_df):
    df = knn_radius_df

    @benchmark
    def run():
        return df.select(
            pds.query_nb_cnt(
                "x1",
                "x2",
                "x3",
                "x4",
                "x5",
                r=_R,
                dist="sql2",
                parallel=True,
            ).alias("cnt")
        )
