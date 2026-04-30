"""
pytest conftest for polars-ds Tier-1 performance benchmarks.

All fixtures are session-scoped so each dataset is constructed once per
pytest invocation and shared across all benchmark functions that request it.
"""

from __future__ import annotations

import pytest
import polars as pl


@pytest.fixture(scope="session")
def glm_irls_df() -> pl.DataFrame:
    """30 M-row (or 3 M-row scaled-down) GLM IRLS group_by workload."""
    from .fixtures import make_glm_irls_df

    return make_glm_irls_df()


@pytest.fixture(scope="session")
def knn_radius_df() -> pl.DataFrame:
    """1 M-row × 5-feature uniform KNN radius query dataset."""
    from .fixtures import make_knn_radius_df

    return make_knn_radius_df()


@pytest.fixture(scope="session")
def entropy_series() -> pl.Series:
    """1 M-element standard-normal f64 Series named 'ts'."""
    from .fixtures import make_entropy_series

    return make_entropy_series()


@pytest.fixture(scope="session")
def rolling_lr_df() -> pl.DataFrame:
    """500 k-row rolling-LR DataFrame with a 100 k-row null prefix on x1..x3."""
    from .fixtures import make_rolling_lr_df

    return make_rolling_lr_df()
