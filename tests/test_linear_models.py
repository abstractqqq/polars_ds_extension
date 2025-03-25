import polars as pl
import polars_ds as pds
import pytest
import numpy as np
from polars_ds.linear_models import OnlineLR, ElasticNet


def test_lr_null_policies_for_np():
    from polars_ds.linear_models import _handle_nans_in_np

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

    x_nan, _ = _handle_nans_in_np(x, y, "ignore")
    assert np.all(np.isnan(x_nan[nulls][:, 0]))

    with pytest.raises(Exception) as exc_info:
        _handle_nans_in_np(x, y, "raise")
        assert str(exc_info.value) == "Nulls found in X or y."

    x_skipped, _ = _handle_nans_in_np(x, y, "skip")
    assert np.all(x_skipped == x[~nulls])

    x_zeroed, _ = _handle_nans_in_np(x, y, "zero")
    assert np.all(
        x_zeroed[nulls][:, 0] == 0.0
    )  # checking out the first column because only that has nulls

    x_one, _ = _handle_nans_in_np(x, y, "one")
    assert np.all(
        x_one[nulls][:, 0] == 1.0
    )  # checking out the first column because only that has nulls


def test_online_lr():
    from sklearn.linear_model import LinearRegression

    size = 5000
    df = (
        pds.frame(size=size)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_row_index()
        .with_columns(
            y=pl.col("x1") + pl.col("x2") * 0.2 - 0.3 * pl.col("x3") + pds.random() * 0.0001
        )
    )
    X = df.select("x1", "x2", "x3").to_numpy()
    y = df.select("y").to_numpy()

    olr = OnlineLR()  # no bias, normal
    sk_lr = LinearRegression(fit_intercept=False)

    olr.fit(X[:10], y[:10])
    coeffs = olr.coeffs()
    sk_lr.fit(X[:10], y[:10])
    sklearn_coeffs = sk_lr.coef_

    pred = olr.predict(X[:10]).flatten()
    sk_pred = sk_lr.predict(X[:10]).flatten()
    assert np.all(np.abs(pred - sk_pred) < 1e-6)
    assert np.all(np.abs(coeffs - sklearn_coeffs) < 1e-6)

    for i in range(10, 20):
        olr.update(X[i], y[i])
        coeffs = olr.coeffs()
        sk_lr = LinearRegression(fit_intercept=False)
        sk_lr.fit(X[: i + 1], y[: i + 1])
        sklearn_coeffs = sk_lr.coef_
        assert np.all(np.abs(coeffs - sklearn_coeffs) < 1e-6)


def _test_elastic_net(add_bias: bool = False):
    import sklearn.linear_model as lm

    l1_reg = 0.1
    l2_reg = 0.1
    alpha = l1_reg + l2_reg
    l1_ratio = l1_reg / (l1_reg + l2_reg)

    df = (
        pds.frame(size=5000)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_row_index()
        .with_columns(
            y=pl.col("x1") + pl.col("x2") * 0.2 - 0.3 * pl.col("x3"),
        )
        .with_columns(is_null=pl.col("x1").is_null())
    )

    X = df.select("x1", "x2", "x3").to_numpy()
    y = df.select("y").to_numpy()
    en = ElasticNet(l1_reg=l1_reg, l2_reg=l2_reg, fit_bias=add_bias)
    elastic = lm.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=add_bias)

    en.fit(X, y)
    pds_res = en.coeffs()
    elastic.fit(X, y)
    sklearn_res = elastic.coef_
    assert np.all(np.abs(pds_res - sklearn_res) < 1e-4)

    if add_bias is True:
        pds_bias = en.bias()
        sklearn_bias = elastic.intercept_
        assert np.all(np.abs(pds_bias - sklearn_bias) < 1e-4)


def test_elastic_net():
    _test_elastic_net(add_bias=False)
    _test_elastic_net(add_bias=True)
