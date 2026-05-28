import polars as pl
import polars_ds as pds
import numpy as np
import pytest
from polars.testing import assert_frame_equal, assert_series_equal


@pytest.mark.parametrize(
    "bias, n",
    [
        (True, 5),
        (True, 10),
        (False, 5),
        (False, 10),
    ],
)
def test_logistic_reg_against_sklearn(bias: bool, n: int):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    # Due to differences in many optimization parameters, we will use a large test set,
    # and accept some small differences in the output
    # Sklearn defaults to a l2 regularized logistic regression, but there is no way to set l2 regularization
    # factor... So won't test l2.
    X, y = make_classification(
        n_samples=10_000,
        n_features=n,
        n_redundant=0,
        n_informative=n - 1,
        random_state=1,
        n_clusters_per_class=1,
    )

    clf = LogisticRegression(
        random_state=0, penalty=None, tol=1e-6, max_iter=400, fit_intercept=bias
    ).fit(X, y)

    variables = [f"x_{i}" for i in range(n)]
    df = pl.from_numpy(X, schema=variables).with_columns(y=y)
    coeffs = df.select(
        pds.logistic_reg(*variables, target="y", add_bias=bias, max_iter=400, tol=1e-6)
    ).item(0, 0)
    coeffs = coeffs.to_numpy()
    test_tol = 1e-5
    if bias:
        assert np.all((coeffs[:-1] - clf.coef_) < test_tol)
        assert abs(coeffs[-1] - clf.intercept_) < test_tol
    else:
        assert np.all((coeffs - clf.coef_) < test_tol)

    y_pred_sklearn = clf.predict_proba(X)[:, 1]
    pred = df.select(
        pds.logistic_reg(
            *variables, target="y", add_bias=bias, max_iter=200, tol=1e-6, return_pred=True
        ).alias("pred")
    )["pred"]
    assert np.all((pred.to_numpy() - y_pred_sklearn) < test_tol)


def test_lin_reg_against_sklearn():
    # Random data + noise
    df = (
        pds.frame(size=5_000)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_columns(
            y=pl.col("x1") * 0.5 + pl.col("x2") * 0.1 - pl.col("x3") * 0.15 + pds.random() * 0.0001
        )
    )

    #
    from sklearn import linear_model

    x = df.select("x1", "x2", "x3").to_numpy()
    y = df["y"].to_numpy()

    # sklearn, normal, with bias
    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit(x, y)

    # pds, normal, with bias
    normal_coeffs = df.select(
        pds.lin_reg(
            "x1",
            "x2",
            "x3",
            target="y",
            add_bias=True,
        ).alias("pred")
    ).explode("pred")
    all_coeffs = normal_coeffs["pred"].to_numpy()
    # non-bias terms, some precision differences expected.
    assert np.all(np.abs(all_coeffs[:3] - reg.coef_) < 1e-5)
    assert np.isclose(all_coeffs[-1], reg.intercept_, rtol=1e-5)

    # sklearn, L2 (Ridge), with bias
    reg = linear_model.Ridge(alpha=0.1, fit_intercept=True)
    reg.fit(x, y)

    # pds, normal, with bias
    normal_coeffs = df.select(
        pds.lin_reg(
            "x1",
            "x2",
            "x3",
            target="y",
            l2_reg=0.1,
            add_bias=True,
        ).alias("pred")
    ).explode("pred")
    all_coeffs = normal_coeffs["pred"].to_numpy()
    coeffs = all_coeffs[:3]
    bias = all_coeffs[-1]
    # non-bias terms, slightly bigger precision differences expected.
    assert np.all(np.abs(coeffs - reg.coef_) < 1e-3)
    assert abs(bias - reg.intercept_) < 1e-3


def test_rolling_ridge():
    # Test rolling linear regression by comparing it with a manually rolled result.
    # Test on multiple window sizes
    size = 500
    df = (
        pds.frame(size=size)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
            pl.Series(name="id", values=list(range(size))),
        )
        .with_columns(
            y=pl.col("x1") * 0.5
            + pl.col("x2") * 0.25
            - pl.col("x3") * 0.15
            + pds.random() * 0.0001,
        )
    )

    for window_size in [5, 8, 12, 15]:
        df_to_test = df.select(
            "id",
            "y",
            pds.rolling_lin_reg(
                "x1",
                "x2",
                "x3",
                target="y",
                l2_reg=0.1,
                window_size=window_size,
            ).alias("result"),
        ).unnest("result")  # .limit(10)
        df_to_test = df_to_test.filter(pl.col("id") >= window_size - 1).select("coeffs")

        results = []
        for i in range(len(df) - window_size + 1):
            temp = df.slice(i, length=window_size)
            results.append(
                temp.select(pds.lin_reg("x1", "x2", "x3", l2_reg=0.1, target="y").alias("coeffs"))
            )

        df_answer = pl.concat(results)
        assert_frame_equal(df_to_test, df_answer)


def test_hc_lin_reg_report():
    import statsmodels.api as sm

    df = (
        pds.frame(size=1000)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
            pds.random(0.0, 1.0).round().cast(pl.UInt8).alias("binary"),
        )
        .with_columns(target=pl.col("x1") * 0.15 + pl.col("x2") * 0.3 + 0.1)
    )

    for se_type in ("se", "hc0", "hc1", "hc2", "hc3"):
        pds_result = df.select(
            pds.lin_reg_report("x1", "x2", "x3", target="target", std_err=se_type).alias("report")
        ).unnest("report")

        Y = df["target"].to_numpy()
        X = df.select("x1", "x2", "x3").to_numpy()

        model = sm.OLS(Y, X)
        results = model.fit()

        if se_type == "se":
            pds_se = pds_result["std_err"].to_numpy()
            sm_se = results.bse
        else:
            pds_se = pds_result[f"{se_type}_se"].to_numpy()
            sm_se = getattr(results, f"{se_type.upper()}_se")

        assert np.all((pds_se - sm_se) < 1e-7)


def test_f32_lin_reg():
    # If they run, they are correct. This is because the underlying functions in src/linalg/ are all
    # generic
    pds.config.LIN_REG_EXPR_F64 = False

    size = 500
    df = (
        pds.frame(size=size)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
            pl.Series(name="id", values=list(range(size))),
        )
        .with_columns(
            y=pl.col("x1") * 0.5
            + pl.col("x2") * 0.25
            - pl.col("x3") * 0.15
            + pds.random() * 0.0001,
            y2=pl.col("x1") + pl.col("x2") * 0.3 - pl.col("x3") * 0.1,
        )
    )

    _ = df.select(
        pds.lin_reg(
            "x1",
            "x2",
            "x3",
            target="y",
        )
    )

    _ = df.select(
        pds.lin_reg(
            "x1",
            "x2",
            "x3",
            target=["y", "y2"],
        )
    )

    _ = df.select(pds.lin_reg("x1", "x2", "x3", target="y", return_pred=True))

    _ = df.select(pds.lin_reg("x1", "x2", "x3", target="y", l1_reg=0.01, return_pred=False))

    _ = df.select(pds.lin_reg("x1", "x2", "x3", target="y", l2_reg=0.01, return_pred=True))

    _ = df.select(
        pds.lin_reg("x1", "x2", "x3", target="y", l1_reg=0.01, l2_reg=0.01, return_pred=False)
    )

    _ = df.select(pds.lin_reg("x1", "x2", "x3", target=["y", "y2"], l2_reg=0.01, return_pred=False))

    _ = df.select(
        pds.lin_reg(
            "x1",
            "x2",
            "x3",
            target=["y", "y2"],
            l2_reg=0.01,
            null_policy="0.1",  # fill 0.1
        )
    )

    _ = df.select(
        pds.lin_reg_report(
            "x1",
            "x2",
            "x3",
            target="y",
        )
    )

    _ = df.select(pds.lin_reg_report("x1", "x2", "x3", target="y", weights="x1"))

    _ = df.select(
        pds.lin_reg_w_rcond(
            "x1",
            "x2",
            "x3",
            target="y",
            rcond=0.3,
        )
    )

    _ = df.select(
        pds.rolling_lin_reg(
            "x1",
            "x2",
            "x3",
            target="y",
            window_size=3,
        ).alias("result"),
    ).unnest("result")

    _ = df.select(
        pds.recursive_lin_reg(
            "x1",
            "x2",
            "x3",
            target="y",
            start_with=3,
        ).alias("result"),
    ).unnest("result")

    # Reset it to true
    pds.config.LIN_REG_EXPR_F64 = True


def test_f32_lin_reg_against_sklearn():
    """f32 path (LIN_REG_EXPR_F64=False) produces coefficients close to sklearn OLS."""
    from sklearn import linear_model

    pds.config.LIN_REG_EXPR_F64 = False
    try:
        df = (
            pds.frame(size=5_000)
            .select(
                pds.random(0.0, 1.0).alias("x1"),
                pds.random(0.0, 1.0).alias("x2"),
                pds.random(0.0, 1.0).alias("x3"),
            )
            .with_columns(
                y=pl.col("x1") * 0.5
                + pl.col("x2") * 0.1
                - pl.col("x3") * 0.15
                + pds.random() * 0.0001
            )
        )

        x = df.select("x1", "x2", "x3").to_numpy()
        y = df["y"].to_numpy()

        # OLS with bias
        reg = linear_model.LinearRegression(fit_intercept=True)
        reg.fit(x, y)
        normal_coeffs = df.select(
            pds.lin_reg("x1", "x2", "x3", target="y", add_bias=True).alias("pred")
        ).explode("pred")
        all_coeffs = normal_coeffs["pred"].to_numpy()
        assert np.all(np.abs(all_coeffs[:3] - reg.coef_) < 1e-4)
        assert abs(all_coeffs[-1] - reg.intercept_) < 1e-4

        # Ridge with bias
        reg_ridge = linear_model.Ridge(alpha=0.1, fit_intercept=True)
        reg_ridge.fit(x, y)
        ridge_coeffs = df.select(
            pds.lin_reg("x1", "x2", "x3", target="y", l2_reg=0.1, add_bias=True).alias("pred")
        ).explode("pred")
        all_ridge = ridge_coeffs["pred"].to_numpy()
        assert np.all(np.abs(all_ridge[:3] - reg_ridge.coef_) < 1e-3)
        assert abs(all_ridge[-1] - reg_ridge.intercept_) < 1e-3

        # Predictions
        pred_result = df.select(
            pds.lin_reg("x1", "x2", "x3", target="y", add_bias=True, return_pred=True).alias(
                "lr_pred"
            )
        ).unnest("lr_pred")
        assert np.all(np.abs(pred_result["pred"].to_numpy() - reg.predict(x)) < 1e-3)

        # lin_reg_report runs without error and returns finite betas in f32 mode
        report = df.select(
            pds.lin_reg_report("x1", "x2", "x3", target="y", add_bias=True).alias("r")
        ).unnest("r")
        betas = np.array(report["beta"].to_list())
        assert np.all(np.isfinite(betas))
        assert np.all(np.abs(betas[:3] - reg.coef_) < 1e-3)
    finally:
        pds.config.LIN_REG_EXPR_F64 = True


def test_pl_lr_multi_pred_correctness():
    """Multi-target return_pred=True predictions match per-target single-target fits."""
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.standard_normal((n, 5))
    true_betas = np.array(
        [
            [0.5, -0.2, 0.1, 0.3, -0.4],
            [0.1, 0.6, -0.3, 0.0, 0.2],
            [-0.2, 0.0, 0.7, -0.1, 0.3],
        ]
    )
    Y = X @ true_betas.T + 0.01 * rng.standard_normal((n, 3))
    df = pl.DataFrame(
        {f"x{i + 1}": X[:, i] for i in range(5)} | {"y1": Y[:, 0], "y2": Y[:, 1], "y3": Y[:, 2]}
    )
    features = [f"x{i + 1}" for i in range(5)]

    multi = df.select(
        pds.lin_reg(*features, target=["y1", "y2", "y3"], return_pred=True).alias("lr_pred")
    ).unnest("lr_pred")

    for i, target in enumerate(["y1", "y2", "y3"]):
        single = df.select(
            pds.lin_reg(*features, target=target, return_pred=True).alias("lr_pred")
        ).unnest("lr_pred")
        np.testing.assert_allclose(
            multi[f"target_{i}_pred"].to_numpy(),
            single["pred"].to_numpy(),
            rtol=0,
            atol=1e-8,
            err_msg=f"multi-pred for target_{i} ({target}) differs from per-target fit",
        )


def test_lin_reg_skip_null():
    df = pl.DataFrame(
        {
            "y": [None, 9.5, 10.5, 11.5, 12.5],
            "a": [1, 9, 10, 11, 12],
            "b": [1.0, 0.5, 0.5, 0.5, 0.5],
        }
    )
    res = pl.DataFrame(
        {
            "pred": [None, 9.5, 10.5, 11.5, 12.5],
            "resid": [None, 0.0, 0.0, 0.0, 0.0],
        }
    )
    assert_frame_equal(
        df.select(
            pds.lin_reg(
                pl.col("a"), pl.col("b"), target="y", return_pred=True, null_policy="skip"
            ).alias("result")
        ).unnest("result"),
        res,
    )


def test_lin_reg_in_group_by():
    df = pl.DataFrame(
        {
            "A": [1] * 4 + [2] * 4,
            "Y": [1] * 8,
            "X1": [1, 2, 3, 4, 5, 6, 7, 8],
            "X2": [2, 3, 4, 1, 6, 7, 8, 5],
        }
    )

    first = df.filter(pl.col("A").eq(1)).with_columns(
        pds.lin_reg(
            pl.col("X1"), pl.col("X2"), target=pl.col("Y"), add_bias=False, return_pred=True
        ).alias("pred")
    )

    second = df.filter(pl.col("A").eq(2)).with_columns(
        pds.lin_reg(
            pl.col("X1"), pl.col("X2"), target=pl.col("Y"), add_bias=False, return_pred=True
        ).alias("pred")
    )

    test = (
        df.group_by("A", maintain_order=True)
        .agg(
            "Y",
            "X1",
            "X2",
            pds.lin_reg(
                pl.col("X1"), pl.col("X2"), target=pl.col("Y"), add_bias=False, return_pred=True
            ).alias("pred"),
        )
        .explode("Y", "X1", "X2", "pred")
    )

    test_first = test.filter(pl.col("A") == 1)
    test_second = test.filter(pl.col("A") == 2)

    assert_frame_equal(first, test_first)
    assert_frame_equal(second, test_second)


def test_lin_reg_with_rcond():
    import numpy as np

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
            y=pl.col("x1") + pl.col("x2") * 0.2 - 0.3 * pl.col("x3"),
        )
    )

    x = df.select("x1", "x2", "x3").to_numpy()
    y = df.select("y").to_numpy()
    np_coeffs, _, _, np_svs = np.linalg.lstsq(x, y, rcond=0.3)  # default rcond
    np_coeffs = np_coeffs.flatten()

    res = df.select(
        pds.lin_reg_w_rcond(
            "x1",
            "x2",
            "x3",
            target="y",
            rcond=0.3,
        ).alias("result")
    ).unnest("result")
    coeffs = res["coeffs"][0].to_numpy()
    svs = res["singular_values"][0].to_numpy()

    assert np.all(np.abs(coeffs - np_coeffs) < 1e-10)
    assert np.all(np.abs(svs - np_svs) < 1e-10)


def test_lasso_regression():
    # These tests have bigger precision tolerance because of different stopping criterions

    from sklearn import linear_model

    df = (
        pds.frame(size=5_000)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_columns(
            y=pl.col("x1") * 0.5 + pl.col("x2") * 0.25 - pl.col("x3") * 0.15 + pds.random() * 0.0001
        )
    )

    x = df.select("x1", "x2", "x3").to_numpy()
    y = df["y"].to_numpy()

    for lambda_ in [0.01, 0.05, 0.1, 0.2]:
        df_res = df.select(
            pds.lin_reg("x1", "x2", "x3", target="y", l1_reg=lambda_, add_bias=False).alias(
                "coeffs"
            )
        ).explode("coeffs")

        res = df_res["coeffs"].to_numpy()

        sklearn = linear_model.Lasso(alpha=lambda_, fit_intercept=False)
        sklearn.fit(x, y)
        res_sklearn = np.asarray(sklearn.coef_)
        assert np.all(np.abs(res_sklearn - res) < 1e-4)

    for lambda_ in [0.01, 0.05, 0.1, 0.2]:
        df_res = df.select(
            pds.lin_reg("x1", "x2", "x3", target="y", l1_reg=lambda_, add_bias=True).alias("coeffs")
        ).explode("coeffs")

        res = df_res["coeffs"].to_numpy()
        res_coef = res[:3]
        res_bias = res[-1]

        sklearn = linear_model.Lasso(alpha=lambda_, fit_intercept=True)
        sklearn.fit(x, y)
        res_sklearn = np.asarray(sklearn.coef_)
        assert np.all(np.abs(res_sklearn - res_coef) < 1e-4)
        assert abs(res_bias - sklearn.intercept_) < 1e-4


def test_positive_lin_reg():
    #
    from sklearn.linear_model import LinearRegression, ElasticNet

    df = (
        pds.frame(size=5_000)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_columns(
            y=pl.col("x1") * 0.5
            + pl.col("x2") * 0.25
            + pl.col("x3") * -0.15
            + pds.random() * 0.0001
        )
    )

    # Third coefficient should be 0 because this is non neg
    for bias in [True, False]:
        pds_result = df.select(
            pds.lin_reg(*[f"x{i + 1}" for i in range(3)], target="y", positive=True, add_bias=bias)
        ).item(0, 0)
        pds_result = pds_result.to_numpy()

        if not bias:
            assert np.all(pds_result >= 0.0)
        else:
            # Bias term can be negative
            assert np.all(pds_result[:-1] >= 0.0)

        reg_nnls = LinearRegression(positive=True, fit_intercept=bias)
        reg_nnls.fit(df.select("x1", "x2", "x3").to_numpy(), df["y"].to_numpy())

        if not bias:
            assert np.all(np.isclose(pds_result, reg_nnls.coef_, atol=1e-5))
        else:
            assert np.all(np.isclose(pds_result[:-1], reg_nnls.coef_, atol=1e-5))
            assert np.isclose(float(pds_result[-1]), reg_nnls.intercept_, atol=1e-5)

    for reg, bias in zip([0.01, 0.05, 0.1, 0.2], [False, True, False, True]):
        l1_reg = reg
        l2_reg = reg

        pds_result = df.select(
            pds.lin_reg(
                "x1", "x2", "x3", target="y", l1_reg=l1_reg, l2_reg=l2_reg, add_bias=bias
            ).alias("coeffs")
        ).item(0, 0)

        pds_result = pds_result.to_numpy()
        if not bias:
            assert np.all(pds_result >= 0.0)
        else:
            # Bias term can be negative
            assert np.all(pds_result[:-1] >= 0.0)

        alpha = l1_reg + l2_reg
        l1_ratio = l1_reg / (l1_reg + l2_reg)

        reg_nnls = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=bias)
        reg_nnls.fit(df.select("x1", "x2", "x3").to_numpy(), df["y"].to_numpy())
        if not bias:  # allow for a higher error tolerance because of convergence differences
            assert np.all(np.isclose(pds_result, reg_nnls.coef_, atol=1e-4))
        else:
            assert np.all(np.isclose(pds_result[:-1], reg_nnls.coef_, atol=1e-4))
            assert np.isclose(float(pds_result[-1]), reg_nnls.intercept_, atol=1e-4)


def test_elastic_net_regression():
    # These tests have bigger precision tolerance because of different stopping criterions

    from sklearn import linear_model

    df = (
        pds.frame(size=5_000)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_columns(
            y=pl.col("x1") * 0.5 + pl.col("x2") * 0.25 - pl.col("x3") * 0.15 + pds.random() * 0.0001
        )
    )

    x = df.select("x1", "x2", "x3").to_numpy()
    y = df["y"].to_numpy()

    for reg in [0.01, 0.05, 0.1, 0.2]:
        l1_reg = reg
        l2_reg = reg

        df_res = df.select(
            pds.lin_reg(
                "x1", "x2", "x3", target="y", l1_reg=l1_reg, l2_reg=l2_reg, add_bias=False
            ).alias("coeffs")
        ).explode("coeffs")

        res = df_res["coeffs"].to_numpy()

        alpha = l1_reg + l2_reg
        l1_ratio = l1_reg / (l1_reg + l2_reg)

        sklearn = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
        sklearn.fit(x, y)
        res_sklearn = np.asarray(sklearn.coef_)
        assert np.all(np.abs(res_sklearn - res) < 1e-4)


def test_recursive_lin_reg():
    # Test against the lstsq method with a fit whenver a new row is in the data
    size = 1_000
    df = (
        pds.frame(size=size)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_columns(
            y=pl.col("x1") * 0.5
            + pl.col("x2") * 0.25
            - pl.col("x3") * 0.15
            + pds.random() * 0.0001,
        )
    )

    start_with = 3

    df_recursive_lr = df.select(
        "y",
        pds.recursive_lin_reg(
            "x1",
            "x2",
            "x3",
            target="y",
            start_with=start_with,
        ).alias("result"),
    ).unnest("result")

    for i in range(start_with, 30):
        coefficients = df.limit(i).select(
            pds.lin_reg(
                "x1",
                "x2",
                "x3",
                target="y",
            ).alias("coeffs")
        )["coeffs"]  # One element series

        normal_result = coefficients[0].to_numpy()
        # i - 1. E.g. use 3 rows of data to train, the data will be at row 2.
        recursive_result = df_recursive_lr["coeffs"][i - 1].to_numpy()
        assert np.all(np.abs(normal_result - recursive_result) < 1e-5)


def test_recursive_ridge():
    # Test against the lstsq method with a fit whenver a new row is in the data
    size = 1_000
    df = (
        pds.frame(size=size)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_columns(
            y=pl.col("x1") * 0.5
            + pl.col("x2") * 0.25
            - pl.col("x3") * 0.15
            + pds.random() * 0.0001,
        )
    )

    start_with = 3

    df_recursive_lr = df.select(
        "y",
        pds.recursive_lin_reg(
            "x1",
            "x2",
            "x3",
            target="y",
            l2_reg=0.1,
            start_with=start_with,
        ).alias("result"),
    ).unnest("result")

    for i in range(start_with, 30):
        coefficients = df.limit(i).select(
            pds.lin_reg(
                "x1",
                "x2",
                "x3",
                target="y",
                l2_reg=0.1,
            ).alias("coeffs")
        )["coeffs"]  # One element series

        normal_result = coefficients[0].to_numpy()
        # i - 1. E.g. use 3 rows of data to train, the data will be at row 2.
        recursive_result = df_recursive_lr["coeffs"][i - 1].to_numpy()
        assert np.all(np.abs(normal_result - recursive_result) < 1e-5)


def test_rolling_lin_reg():
    # Test rolling linear regression by comparing it with a manually rolled result.
    # Test on multiple window sizes
    size = 500
    df = (
        pds.frame(size=size)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
            pl.Series(name="id", values=list(range(size))),
        )
        .with_columns(
            y=pl.col("x1") * 0.5
            + pl.col("x2") * 0.25
            - pl.col("x3") * 0.15
            + pds.random() * 0.0001,
        )
    )

    for window_size in [5, 8, 12, 15]:
        df_to_test = df.select(
            "id",
            "y",
            pds.rolling_lin_reg(
                "x1",
                "x2",
                "x3",
                target="y",
                window_size=window_size,
            ).alias("result"),
        ).unnest("result")  # .limit(10)
        df_to_test = df_to_test.filter(pl.col("id") >= window_size - 1).select("coeffs")

        results = []
        for i in range(len(df) - window_size + 1):
            temp = df.slice(i, length=window_size)
            results.append(temp.select(pds.lin_reg("x1", "x2", "x3", target="y").alias("coeffs")))

        df_answer = pl.concat(results)
        assert_frame_equal(df_to_test, df_answer)


# This only tests that nulls are correctly skipped.
def test_rolling_null_skips():
    size = 1000
    # Data with random nulls
    df = (
        pds.frame(size=size)
        .select(
            pds.random(0.0, 1.0).alias("x1"),
            pds.random(0.0, 1.0).alias("x2"),
            pds.random(0.0, 1.0).alias("x3"),
        )
        .with_columns(
            pl.when(pds.random() < 0.15).then(None).otherwise(pl.col("x1")).alias("x1"),
            pl.when(pds.random() < 0.15).then(None).otherwise(pl.col("x2")).alias("x2"),
            pl.when(pds.random() < 0.15).then(None).otherwise(pl.col("x3")).alias("x3"),
        )
        .with_columns(
            null_ref=pl.any_horizontal(
                pl.col("x1").is_null(), pl.col("x2").is_null(), pl.col("x3").is_null()
            ),
            y=pl.col("x1") * 0.15 + pl.col("x2") * 0.3 - pl.col("x3") * 1.5 + pds.random() * 0.0001,
        )
    )

    window_size = 6
    min_valid_rows = 5

    result = df.with_columns(
        pds.rolling_lin_reg(
            "x1",
            "x2",
            "x3",
            target="y",
            window_size=window_size,
            min_valid_rows=min_valid_rows,
            null_policy="skip",
        ).alias("test")
    ).with_columns(
        pl.col("test").struct.field("coeffs").alias("coeffs"),
        pl.col("test").struct.field("coeffs").is_null().alias("is_null"),
    )

    nulls = df["null_ref"].to_list()  # list of bools
    rolling_should_be_null = [True] * (window_size - 1)
    for i in range(0, len(nulls) - window_size + 1):
        lower = i
        upper = i + window_size
        window_valid_count = window_size - np.sum(nulls[lower:upper])  # size - null count
        rolling_should_be_null.append((window_valid_count < min_valid_rows))

    answer = pl.Series(name="is_null", values=rolling_should_be_null)
    assert_series_equal(result["is_null"], answer)


# ---------------------------------------------------------------------------
# Tests covering the perf refactor of `series_to_mat_for_lr`, the cast guards
# in lin_reg_report / wls_report, the StructChunked multi-target output, and
# the unchecked series_to_slice fast path.
# ---------------------------------------------------------------------------


def test_lin_reg_many_small_groups_matches_per_group():
    """
    Exercises the small-group group_by_agg path (Pattern A in the PR).
    Builds 200 groups × 25 rows and asserts that the coefficients produced
    via group_by_agg match a per-group fit run independently.
    """
    rng = np.random.default_rng(0)
    n_groups, n_per = 200, 25
    gids = np.repeat(np.arange(n_groups), n_per)
    df = pl.DataFrame(
        {
            "g": gids,
            "x1": rng.standard_normal(len(gids)),
            "x2": rng.standard_normal(len(gids)),
            "y": rng.standard_normal(len(gids)),
        }
    )

    grouped = (
        df.group_by("g", maintain_order=True)
        .agg(pds.lin_reg("x1", "x2", target="y", add_bias=True).alias("coef"))
        .sort("g")
    )

    expected = []
    for g in range(n_groups):
        sub = df.filter(pl.col("g") == g)
        coef = sub.select(
            pds.lin_reg("x1", "x2", target="y", add_bias=True).alias("coef")
        ).row(0)[0]
        expected.append(coef)

    got = grouped["coef"].to_list()
    assert len(got) == len(expected)
    for g_coef, e_coef in zip(got, expected):
        np.testing.assert_allclose(g_coef, e_coef, rtol=1e-12, atol=1e-12)


def test_lin_reg_with_bias_appended_column_equivalence():
    """
    The refactor writes the bias column directly into the design matrix
    instead of going through `df.lazy().with_column(lit(1.0)).collect()`.
    Confirm `add_bias=True` produces the same result as `add_bias=False`
    with a manually-appended unit column.
    """
    rng = np.random.default_rng(1)
    n = 500
    df = pl.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y": rng.standard_normal(n),
            "ones": np.ones(n),
        }
    )

    with_flag = df.select(
        pds.lin_reg("x1", "x2", target="y", add_bias=True).alias("coef")
    ).row(0)[0]
    manual = df.select(
        pds.lin_reg("x1", "x2", "ones", target="y", add_bias=False).alias("coef")
    ).row(0)[0]

    np.testing.assert_allclose(with_flag, manual, rtol=1e-10, atol=1e-12)


def test_lin_reg_report_already_float64_cast_guard():
    """
    `pl_lin_reg_report` skips the cast when inputs are already Float64.
    Verify the cast-skip Float64 path still produces correct results
    (matches a NumPy OLS reference) and matches a path forced through
    cast via Float32 inputs.
    """
    rng = np.random.default_rng(2)
    n = 300
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 0.5 * x1 - 0.3 * x2 + 0.1 * rng.standard_normal(n)

    df_f64 = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    df_f32 = df_f64.with_columns(
        [
            pl.col("x1").cast(pl.Float32),
            pl.col("x2").cast(pl.Float32),
            pl.col("y").cast(pl.Float32),
        ]
    )

    rep_f64 = df_f64.select(
        pds.lin_reg_report("x1", "x2", target="y", add_bias=True).alias("r")
    ).unnest("r")
    # Float32 inputs force the cast branch in pl_lin_reg_report (Float64-target
    # cast actually executes), exercising the non-skip code path.
    rep_f32 = df_f32.select(
        pds.lin_reg_report("x1", "x2", target="y", add_bias=True).alias("r")
    ).unnest("r")

    # Same coefficients (Float32 → Float64 cast is exact for these values).
    np.testing.assert_allclose(
        rep_f64["beta"].to_list(),
        rep_f32["beta"].to_list(),
        rtol=1e-6,
        atol=1e-7,
    )

    # Independent NumPy reference for the Float64 path.
    X = np.column_stack([x1, x2, np.ones(n)])
    beta_ref, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(
        rep_f64["beta"].to_list(), beta_ref, rtol=1e-10, atol=1e-12
    )


def test_wls_report_multichunked_weights_dont_panic():
    """
    The cast guard in `pl_wls_report` requires `n_chunks() == 1` for the
    weights series before it skips the cast. Pass a multi-chunk Float64
    weights series and confirm the guard correctly falls through to the
    cast (rechunk) branch instead of panicking on `cont_slice().unwrap()`.
    """
    rng = np.random.default_rng(3)
    n = 200
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.1 * rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)

    # Force multi-chunk weights by concatenating two halves.
    w_part1 = pl.Series("w", w[: n // 2])
    w_part2 = pl.Series("w", w[n // 2 :])
    w_multi = pl.concat([w_part1, w_part2], rechunk=False)
    assert w_multi.n_chunks() == 2

    df = pl.DataFrame({"x": x, "y": y}).with_columns(w_multi)
    rep = df.select(
        pds.lin_reg_report("x", target="y", weights="w", add_bias=True).alias("r")
    ).unnest("r")

    # Just verify it ran, produced finite coefficients, and matches a
    # single-chunk run (which goes down the cast-skip branch).
    df_single = df.with_columns(pl.col("w").rechunk())
    rep_single = df_single.select(
        pds.lin_reg_report("x", target="y", weights="w", add_bias=True).alias("r")
    ).unnest("r")
    np.testing.assert_allclose(
        rep["beta"].to_list(),
        rep_single["beta"].to_list(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_lin_reg_multi_target_struct_output():
    """
    Exercise the StructChunked::from_columns multi-target output path
    (Pattern B in the PR) on a single frame. Multi-target lin_reg inside
    group_by_agg has a separate Polars-side schema-mismatch limitation
    that pre-dates this PR; this test stays in `select(...)` to isolate
    the StructChunked output construction.
    """
    rng = np.random.default_rng(4)
    n = 1000
    df = pl.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y1": rng.standard_normal(n),
            "y2": rng.standard_normal(n),
        }
    )

    multi = df.select(
        pds.lin_reg(
            "x1", "x2", target=["y1", "y2"], add_bias=True
        ).alias("coef")
    )
    s_multi = multi["coef"].to_list()[0]

    # Per-target single-target fits should match the corresponding field
    # of the multi-target struct output.
    coef_y1 = df.select(
        pds.lin_reg("x1", "x2", target="y1", add_bias=True).alias("c")
    ).row(0)[0]
    coef_y2 = df.select(
        pds.lin_reg("x1", "x2", target="y2", add_bias=True).alias("c")
    ).row(0)[0]

    # Field order in the struct must be preserved by
    # StructChunked::from_columns (mirrors the original DataFrame::new arg
    # order). Field names follow the existing positional convention
    # `target_0`, `target_1`, ...
    keys = list(s_multi.keys())
    assert keys == ["target_0", "target_1"], (
        f"unexpected struct field order: {keys}"
    )
    np.testing.assert_allclose(s_multi["target_0"], coef_y1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s_multi["target_1"], coef_y2, rtol=1e-12, atol=1e-12)


def test_lin_reg_single_big_fit_no_regression_path():
    """
    Sanity check the non-small-group path (single fit, n above the rayon
    threshold). The size gate must NOT change numeric output — this test
    only verifies bit-stable coefficients vs an sklearn reference, ensuring
    the gated parallel path produces the same numbers as before.
    """
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(5)
    n, p = 50_000, 6  # comfortably above PARALLEL_MATMUL_THRESHOLD (4096)
    X = rng.standard_normal((n, p))
    true_beta = np.array([0.4, -0.2, 0.7, 0.0, -0.1, 0.3])
    y = X @ true_beta + 0.05 * rng.standard_normal(n)

    df = pl.DataFrame(
        {f"x{i}": X[:, i] for i in range(p)} | {"y": y}
    )
    pds_coef = df.select(
        pds.lin_reg(*[f"x{i}" for i in range(p)], target="y", add_bias=True).alias("c")
    ).row(0)[0]

    sk = LinearRegression(fit_intercept=True).fit(X, y)
    sk_coef = list(sk.coef_) + [sk.intercept_]

    np.testing.assert_allclose(pds_coef, sk_coef, rtol=1e-8, atol=1e-10)


def test_lin_reg_null_skip_in_small_group():
    """
    Combined: many-small-groups path AND null rows. The no-null fast path
    must NOT be taken; the null-handling code path must still produce the
    correct skip-null behavior.
    """
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "x": [1.0, 2.0, None, 4.0, 1.0, None, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, None, 0.5, 1.0, 1.5, 2.0],
        }
    )
    out = (
        df.group_by("g", maintain_order=True)
        .agg(
            pds.lin_reg("x", target="y", add_bias=True, null_policy="skip").alias("c")
        )
        .sort("g")
    )

    # Reference: drop nulls per group then fit.
    expected = []
    for g in [1, 2, 3]:
        sub = df.filter(pl.col("g") == g).drop_nulls()
        coef = sub.select(
            pds.lin_reg("x", target="y", add_bias=True).alias("c")
        ).row(0)[0]
        expected.append(coef)

    for got, exp in zip(out["c"].to_list(), expected):
        np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# singular_x_tol: rank-deficiency gate (issue #461)
# ---------------------------------------------------------------------------

import polars_ds.config as _pds_cfg


@pytest.fixture(params=["f64", "f32"])
def lin_reg_dtype(request, monkeypatch):
    """Run each gate test under both the f64 and f32 plugin variants."""
    monkeypatch.setattr(_pds_cfg, "LIN_REG_EXPR_F64", request.param == "f64")
    return request.param


def _collinear_df(n: int = 64, seed: int = 0) -> pl.DataFrame:
    """A design whose X'X is exactly singular (x2 = 2*x1)."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "x1": x1,
            "x2": 2.0 * x1,  # perfectly collinear -> rank-deficient X'X
            "y": rng.standard_normal(n),
        }
    )


def test_singular_x_tol_nulls_collinear_coeffs(lin_reg_dtype):
    # Default singular_x_tol (1e-12) gates the singular design to null.
    df = _collinear_df()
    out = df.select(pds.lin_reg("x1", "x2", target="y", add_bias=False).alias("coeffs"))
    assert out["coeffs"][0] is None


def test_singular_x_tol_off_returns_finite(lin_reg_dtype):
    # singular_x_tol=0.0 disables the gate -> finite (min-norm) solution as before.
    df = _collinear_df()
    out = df.select(
        pds.lin_reg("x1", "x2", target="y", add_bias=False, singular_x_tol=0.0).alias("coeffs")
    )
    coeffs = out["coeffs"][0]
    assert coeffs is not None
    assert len(coeffs.to_list()) == 2


def test_singular_x_tol_well_conditioned_unchanged(lin_reg_dtype):
    # A well-conditioned design must NOT be gated and must match sklearn.
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(7)
    n = 500
    df = pl.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
        }
    ).with_columns(y=pl.col("x1") - 0.5 * pl.col("x2") + 2.0 * pl.col("x3"))

    coeffs = df.select(
        pds.lin_reg("x1", "x2", "x3", target="y", add_bias=False).alias("c")
    )["c"][0]
    assert coeffs is not None

    X = df.select("x1", "x2", "x3").to_numpy()
    y = df["y"].to_numpy()
    ref = LinearRegression(fit_intercept=False).fit(X, y).coef_
    tol = 1e-4 if lin_reg_dtype == "f32" else 1e-9
    np.testing.assert_allclose(coeffs.to_list(), ref, rtol=tol, atol=tol)


def test_singular_x_tol_group_by_nulls_degenerate_group(lin_reg_dtype):
    # The use case: one degenerate group nulls, the good group fits.
    rng = np.random.default_rng(11)
    n = 40
    good_x1 = rng.standard_normal(n)
    bad_x1 = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "g": ["good"] * n + ["bad"] * n,
            "x1": np.concatenate([good_x1, bad_x1]),
            "x2": np.concatenate([rng.standard_normal(n), 2.0 * bad_x1]),  # bad group collinear
            "y": rng.standard_normal(2 * n),
        }
    )
    res = df.group_by("g", maintain_order=True).agg(
        pds.lin_reg("x1", "x2", target="y", add_bias=False).alias("c")
    )
    cmap = {row[0]: row[1] for row in res.iter_rows()}
    assert cmap["bad"] is None
    assert cmap["good"] is not None


def test_singular_x_tol_return_pred_nulls(lin_reg_dtype):
    # return_pred path: a gated group yields all-null pred & resid.
    df = _collinear_df(n=32)
    out = df.select(
        pds.lin_reg("x1", "x2", target="y", add_bias=False, return_pred=True).alias("p")
    ).unnest("p")
    assert out["pred"].null_count() == df.height
    assert out["resid"].null_count() == df.height


def test_singular_x_tol_multi_target_nulls(lin_reg_dtype):
    # multi-target coeffs path: all target fields null on a gated design.
    df = _collinear_df(n=64).with_columns(y2=pl.col("y") * 0.5 + 1.0)
    s = df.select(
        pds.lin_reg("x1", "x2", target=["y", "y2"], add_bias=False).alias("c")
    )["c"].to_list()[0]
    assert s["target_0"] is None
    assert s["target_1"] is None
