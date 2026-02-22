import polars as pl
import polars_ds as pds
import numpy as np
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

@pytest.mark.parametrize(
    "bias, n",
    [
        (True, 5), (True, 10),
        (False, 5), (False, 10),
    ],
)
def test_logistic_reg_against_sklearn(bias:bool, n:int):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    
    # Due to differences in many optimization parameters, we will use a large test set,
    # and accept some small differences in the output
    # Sklearn defaults to a l2 regularized logistic regression, but there is no way to set l2 regularization
    # factor... So won't test l2.
    X, y = make_classification(n_samples=10_000, n_features=n, n_redundant=0, 
                        n_informative=n-1, random_state=1, n_clusters_per_class=1)

    clf = LogisticRegression(random_state=0, penalty = None, tol = 1e-6, max_iter=400, fit_intercept=bias).fit(X, y)

    variables = [f"x_{i}" for i in range(n)]
    df = pl.from_numpy(X, schema=variables).with_columns(y = y)
    coeffs = df.select(
        pds.logistic_reg(
            *variables
            , target = 'y'
            , add_bias = bias
            , max_iter = 400
            , tol = 1e-6
        )
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
            *variables
            , target = 'y'
            , add_bias = bias
            , max_iter = 200
            , tol = 1e-6
            , return_pred = True
        ).alias('pred')
    )['pred']
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