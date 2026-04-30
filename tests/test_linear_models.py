import polars as pl
import polars_ds as pds
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from polars_ds.linear_models import OnlineLR, ElasticNet, LR


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


@pytest.mark.parametrize("solver", ["svd", "cholesky", "qr"])
def test_lr(solver):
    ols = LR(False, 0.0, solver)
    df = (
        pds.frame(size=5000)
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

    ols.fit(X, y)
    coeffs = ols.coeffs()
    sk_ols = LinearRegression(fit_intercept=False)
    sk_ols.fit(X, y)
    sklearn_coeffs = sk_ols.coef_
    assert np.all(np.abs(coeffs - sklearn_coeffs) < 1e-6)


def test_online_lr():
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
    en = ElasticNet(l1_reg=l1_reg, l2_reg=l2_reg, has_bias=add_bias)
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


def test_glm():
    import statsmodels.api as sm
    from polars_ds.linear_models import GLM

    data = sm.datasets.scotland.load()
    # add_bias = True

    data.exog = sm.add_constant(data.exog)
    df = data.exog
    df["y"] = data.endog
    # X1 contains constant term, X2 doesn't
    X1 = df[[c for c in df.columns if c != "y"]].to_numpy()
    X2 = df[[c for c in df.columns if c not in ("y", "const")]].to_numpy()
    y = df["y"].to_numpy()

    glm = sm.GLM(y, X1, family=sm.families.Gamma())
    glm_results = glm.fit()
    params = glm_results.params
    sm_bias = params[0]
    sm_coeffs = params[1:]

    pds_glm = GLM(solver="irls", add_bias=True, family="gamma")

    pds_glm.fit(X2, y.reshape((-1, 1)))

    pds_bias = pds_glm.bias()
    pds_coeffs = pds_glm.coeffs()

    assert abs(sm_bias - pds_bias) < 1e-6
    np.testing.assert_allclose(sm_coeffs, pds_coeffs, rtol=1e-5)


@pytest.mark.parametrize(
    "family,sm_family_cls,y_gen",
    [
        # gaussian: identity link, y ~ N(eta, 0.1)
        (
            "gaussian",
            "Gaussian",
            lambda X, rng: X @ np.array([1.0, -0.5, 0.3, 0.8]) + rng.randn(X.shape[0]) * 0.1,
        ),
        # binomial: logit link, y ~ Bernoulli(sigmoid(eta))
        (
            "binomial",
            "Binomial",
            lambda X, rng: rng.binomial(
                1,
                1.0 / (1.0 + np.exp(-(X @ np.array([1.0, -0.5, 0.3, 0.8])))),
            ).astype(float),
        ),
        # poisson: log link, y ~ Poisson(exp(eta))  — clip eta to keep rates finite
        (
            "poisson",
            "Poisson",
            lambda X, rng: rng.poisson(
                np.exp(np.clip(X @ np.array([0.5, -0.25, 0.15, 0.4]), -2.0, 2.0))
            ).astype(float),
        ),
        # gamma: inverse link (canonical), y positive from Gamma distribution
        (
            "gamma",
            "Gamma",
            lambda X, rng: rng.gamma(
                shape=2.0,
                scale=np.exp(np.clip(X @ np.array([0.3, -0.15, 0.09, 0.24]), -2.0, 2.0)) / 2.0,
            ),
        ),
    ],
)
def test_glm_family(family, sm_family_cls, y_gen):
    """
    Parametrized test for all four GLM families: gaussian, binomial, poisson, gamma.
    Each family is compared against statsmodels GLM using the same canonical link function,
    so the expected coefficients should match to atol=1e-2 (loose tolerance because IRLS
    implementations may use slightly different weight initialisation or stopping criteria).

    statsmodels families and their canonical links used here:
      gaussian  -> identity
      binomial  -> logit
      poisson   -> log
      gamma     -> inverse (default in statsmodels)
    """
    import statsmodels.api as sm
    from polars_ds.linear_models import GLM

    rng = np.random.RandomState(42)
    n, p = 500, 4
    X = rng.randn(n, p)
    y = y_gen(X, rng)

    # statsmodels GLM with bias column prepended to X
    X_with_const = sm.add_constant(X)
    sm_family = getattr(sm.families, sm_family_cls)()
    sm_glm = sm.GLM(y, X_with_const, family=sm_family).fit()
    sm_bias = sm_glm.params[0]
    sm_coeffs = sm_glm.params[1:]

    # pds GLM with add_bias=True
    pds_glm = GLM(solver="irls", add_bias=True, family=family, max_iter=200, tol=1e-8)
    pds_glm.fit(X, y.reshape(-1, 1))
    pds_coeffs = pds_glm.coeffs()
    pds_bias = pds_glm.bias()

    np.testing.assert_allclose(
        pds_coeffs,
        sm_coeffs,
        atol=1e-2,
        err_msg=f"GLM family={family!r}: coefficients differ from statsmodels by more than atol=1e-2",
    )
    assert abs(pds_bias - sm_bias) < 1e-2, (
        f"GLM family={family!r}: bias {pds_bias} differs from statsmodels {sm_bias} by more than 1e-2"
    )


def test_glm_convergence_failure():
    """
    Verify that GLM with max_iter=1 and very tight tol=1e-12 does not crash.
    The IRLS implementation prints a non-convergence message to stdout but still
    returns coefficients (best iterate after 1 step). This test documents that
    behaviour: no exception is raised and coefficients are finite.
    """
    from polars_ds.linear_models import GLM

    rng = np.random.RandomState(0)
    n, p = 200, 5
    X = rng.randn(n, p)
    y = np.exp(np.clip(X @ np.ones(p) * 0.2, -2, 2)) + rng.exponential(0.5, n)
    y = y.reshape(-1, 1)

    # max_iter=1 guarantees non-convergence on any non-trivial problem
    glm = GLM(solver="irls", add_bias=False, family="poisson", max_iter=1, tol=1e-12)
    # Must not raise
    glm.fit(X, y)

    coeffs = glm.coeffs()
    # Coefficients should exist and be finite (not NaN/Inf)
    assert coeffs is not None
    assert np.all(np.isfinite(coeffs)), f"Expected finite coeffs after 1 IRLS step, got {coeffs}"
