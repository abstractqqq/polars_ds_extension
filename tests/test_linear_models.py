import polars as pl
import polars_ds as pds
import pytest
import numpy as np
from polars_ds.linear_models import LR


def test_lr_null_policies_for_np():
    size = 5_000
    df = (
        pds.frame(size=size)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_row_index()
        .with_columns(
            x1=pl.when(pl.col("x1") > 0.5).then(None).otherwise(pl.col("x1")),
            y=pl.col("x1") + pl.col("x2") * 0.2 - 0.3 * pl.col("x3"),
        )
        .with_columns(is_null=pl.col("x1").is_null())
    )
    nulls = df.select("is_null").to_numpy().flatten()
    x = df.select("x1", "x2", "x3").to_numpy()
    y = df.select("y").to_numpy()

    x_nan, _ = LR._handle_nans_in_np(x, y, "ignore")
    assert np.all(np.isnan(x_nan[nulls][:, 0]))

    with pytest.raises(Exception) as exc_info:
        LR._handle_nans_in_np(x, y, "raise")
        assert str(exc_info.value) == "Nulls found in X or y."

    x_skipped, _ = LR._handle_nans_in_np(x, y, "skip")
    assert np.all(x_skipped == x[~nulls])

    x_zeroed, _ = LR._handle_nans_in_np(x, y, "zero")
    assert np.all(
        x_zeroed[nulls][:, 0] == 0.0
    )  # checking out the first column because only that has nulls

    x_one, _ = LR._handle_nans_in_np(x, y, "one")
    assert np.all(
        x_one[nulls][:, 0] == 1.0
    )  # checking out the first column because only that has nulls
