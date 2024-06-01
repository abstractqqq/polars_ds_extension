from __future__ import annotations

import pytest
import polars as pl
import numpy as np
import polars_ds as pds
from polars.testing import assert_frame_equal


def test_pca():
    from sklearn.decomposition import PCA

    df = pds.random_data(size=2000, n_cols=0).select(
        pds.random(0.0, 1.0).alias("x1"),
        pds.random(0.0, 1.0).alias("x2"),
        pds.random(0.0, 1.0).alias("x3"),
    )

    singular_values = df.select(pds.query_singular_values("x1", "x2", "x3").alias("res"))["res"][
        0
    ].to_numpy()

    pca = PCA()
    data_matrix = df.select("x1", "x2", "x3").to_numpy().astype(np.float64)
    pca.fit(data_matrix)
    ans_singular_values = pca.singular_values_

    assert np.isclose(singular_values, ans_singular_values).all()

    singular_values = df.select(pds.query_singular_values("x1", "x2", "x3").alias("res"))["res"][
        0
    ].to_numpy()

    vectors = df.select(pds.query_pca("x1", "x2", "x3").alias("vectors")).unnest("vectors")[
        "weight_vector"
    ]

    ans_vectors = pca.components_
    for i in range(len(vectors)):
        vi = vectors[i].to_numpy()
        ans_vi = ans_vectors[i, :]
        # The principal vector can be either v or -v. It doesn't matter and depends on backend algo's choice.
        # Have a more relaxed rtol
        assert np.isclose(np.abs(vi), np.abs(ans_vi), rtol=1e-5).all()


def test_copula_entropy():
    from numpy.random import multivariate_normal as mnorm
    import copent

    rho = 0.6
    mean1 = [0, 0]
    cov1 = [[1, rho], [rho, 1]]
    x = mnorm(mean1, cov1, 200)  # bivariate gaussian
    ce1 = copent.copent(x, dtype="euclidean")  # estimated copula entropy

    df = pl.from_numpy(x, schema=["x1", "x2"])
    res = df.select(pds.query_copula_entropy("x1", "x2", k=3)).item(0, 0)

    assert np.isclose(res, ce1)


def test_cond_indep_and_transfer():
    from copent import ci, transent

    df = pds.random_data(size=2_000, n_cols=0).select(
        pds.random(0.0, 1.0).alias("x1"),
        pds.random(0.0, 1.0).alias("x2"),
        pds.random(0.0, 1.0).alias("x3"),
    )

    ci_ans = ci(df["x1"].to_numpy(), df["x2"].to_numpy(), df["x3"].to_numpy(), dtype="euclidean")
    ci_res = df.select(pds.query_cond_indep("x1", "x2", "x3", k=3)).item(0, 0)

    assert np.isclose(ci_ans, ci_res)

    t_ans = transent(df["x1"].to_numpy(), df["x2"].to_numpy(), dtype="euclidean")
    t_res = df.select(pds.query_transfer_entropy("x1", "x2", k=3)).item(0, 0)

    assert np.isclose(t_ans, t_res)


def test_xi_corr():
    df = pds.random_data(size=2_000, n_cols=0).select(
        pds.random(0.0, 12.0).alias("x"),
        pds.random(0.0, 1.0).alias("y"),
    )

    from xicor.xicor import Xi

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    xi_obj = Xi(x, y)
    ans_statistic = xi_obj.correlation
    test_statistic = df.select(pds.xi_corr("x", "y")).item(0, 0)

    assert np.isclose(ans_statistic, test_statistic, rtol=1e-4)


def test_kendall_tau():
    from scipy.stats import kendalltau

    df = pds.random_data(size=2000, n_cols=0).select(
        pds.random_int(0, 200).alias("x"),
        pds.random_int(0, 200).alias("y"),
    )

    test = df.select(pds.kendall_tau("x", "y")).item(0, 0)

    res = kendalltau(df["x"].to_numpy(), df["y"].to_numpy())

    assert np.isclose(test, res.statistic)


@pytest.mark.parametrize(
    "df, ft, res_full, res_valid, res_same",
    [
        (
            pl.DataFrame({"a": [5, 6, 7, 8, 9]}),
            [1, 0, -1],
            pl.DataFrame({"a": pl.Series([5, 6, 2, 2, 2, -8, -9], dtype=pl.Float64)}),
            pl.DataFrame({"a": pl.Series([2, 2, 2], dtype=pl.Float64)}),
            pl.DataFrame({"a": pl.Series([6, 2, 2, 2, -8], dtype=pl.Float64)}),
        ),
    ],
)
def test_convolve(df, ft, res_full, res_valid, res_same):
    res = df.select(pds.convolve("a", ft, mode="full"))

    assert_frame_equal(res, res_full)

    res = df.select(pds.convolve("a", ft, mode="valid"))

    assert_frame_equal(res, res_valid)

    res = df.select(pds.convolve("a", ft, mode="same"))

    assert_frame_equal(res, res_same)

    res = df.select(pds.convolve("a", ft, mode="full", parallel=True))

    assert_frame_equal(res, res_full)

    res = df.select(pds.convolve("a", ft, mode="valid", parallel=True))

    assert_frame_equal(res, res_valid)

    res = df.select(pds.convolve("a", ft, mode="same", parallel=True))

    assert_frame_equal(res, res_same)

    res = df.select(pds.convolve("a", ft, mode="full", method="fft"))

    assert_frame_equal(res, res_full)

    res = df.select(pds.convolve("a", ft, mode="valid", method="fft"))

    assert_frame_equal(res, res_valid)

    res = df.select(pds.convolve("a", ft, mode="same", method="fft"))

    assert_frame_equal(res, res_same)


@pytest.mark.parametrize(
    "arr, n",
    [
        # n is Optional[int]
        (np.random.normal(size=200), None),
        (np.random.normal(size=200), 100),
    ],
)
def test_fft(arr, n):
    import scipy as sp

    df = pl.DataFrame({"a": arr})
    res = df.select(pds.rfft("a", n=n).alias("fft")).select(
        pl.col("fft").arr.first().alias("re"), pl.col("fft").arr.last().alias("im")
    )
    real_test = res["re"].to_numpy()
    im_test = res["im"].to_numpy()

    ans = np.fft.rfft(arr, n=n)
    real = ans.real
    imag = ans.imag
    assert np.isclose(real_test, real).all()
    assert np.isclose(im_test, imag).all()

    # Always run a check against scipy, with return_full=True as well
    res2 = df.select(pds.rfft("a", return_full=True).alias("fft")).select(
        pl.col("fft").arr.first().alias("re"), pl.col("fft").arr.last().alias("im")
    )
    real_test = res2["re"].to_numpy()
    im_test = res2["im"].to_numpy()
    ans = sp.fft.fft(arr)  # always full fft
    real = ans.real
    imag = ans.imag
    assert np.isclose(real_test, real).all()
    assert np.isclose(im_test, imag).all()


@pytest.mark.parametrize(
    "df",
    [
        (
            pl.DataFrame(
                {
                    "target": np.random.randint(0, 3, size=1_000),
                    "a": np.random.normal(size=1_000),
                }
            )
        ),
    ],
)
def test_f_test(df):
    from sklearn.feature_selection import f_classif

    res = df.select(pds.query_f_test(pl.col("a"), group=pl.col("target")))
    res = res.item(0, 0)  # A dictionary
    statistic = res["statistic"]
    pvalue = res["pvalue"]

    scikit_res = f_classif(df["a"].to_numpy().reshape(-1, 1), df["target"].to_numpy())
    scikit_s = scikit_res[0][0]
    scikit_p = scikit_res[1][0]

    assert np.isclose(statistic, scikit_s)
    assert np.isclose(pvalue, scikit_p)


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": list(range(24))}),
            [11, 5, 1, 1, 1, 1, 1, 1, 1],
        ),
        (
            pl.DataFrame({"a": [1, 2, 3, 4, float("nan"), float("inf"), None]}),
            [1, 1, 1, 1, 0, 0, 0, 0, 0],  # NaN, Inf, None are ignored
        ),
    ],
)
def test_first_digit_cnt(df, res):
    assert_frame_equal(
        df.select(pds.query_first_digit_cnt("a").explode().cast(pl.UInt32)),
        pl.DataFrame({"a": pl.Series(values=res, dtype=pl.UInt32)}),
    )
    assert_frame_equal(
        df.lazy().select(pds.query_first_digit_cnt("a").explode().cast(pl.UInt32)).collect(),
        pl.DataFrame({"a": pl.Series(values=res, dtype=pl.UInt32)}),
    )


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": [2.123, None, -2.111, float("nan")]}),
            pl.DataFrame({"a": [2.0, None, -2.0, float("nan")]}),
        ),
    ],
)
def test_trunc(df, res):
    assert_frame_equal(df.select(pds.trunc("a")), res)
    assert_frame_equal(df.lazy().select(pds.trunc("a")).collect(), res)


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": [2.123, None, -2.111, float("nan")]}),
            pl.DataFrame({"a": [0.123, None, -0.111, float("nan")]}),
        ),
    ],
)
def test_fract(df, res):
    assert_frame_equal(df.select(pds.fract("a")), res)
    assert_frame_equal(df.lazy().select(pds.fract("a")).collect(), res)


@pytest.mark.parametrize(
    "df, other, res",
    [
        (
            pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 2, 2, 10]}),
            3,
            pl.DataFrame({"a": [1, 1, 3, 1, 1]}),
        ),
        (
            pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 2, 2, 10]}),
            pl.col("b"),
            pl.DataFrame({"a": [1, 2, 1, 2, 5]}),
        ),
    ],
)
def test_gcd(df, other, res):
    assert_frame_equal(df.select(pds.query_gcd("a", other).cast(pl.Int64)), res)

    assert_frame_equal(df.lazy().select(pds.query_gcd("a", other).cast(pl.Int64)).collect(), res)


@pytest.mark.parametrize(
    "df, other, res",
    [
        (
            pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 2, 2, 10]}),
            3,
            pl.DataFrame({"a": [3, 6, 3, 12, 15]}),
        ),
        (
            pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 2, 2, 10]}),
            pl.col("b"),
            pl.DataFrame({"a": [1, 2, 6, 4, 10]}),
        ),
    ],
)
def test_lcm(df, other, res):
    assert_frame_equal(df.select(pds.query_lcm("a", other).cast(pl.Int64)), res)

    assert_frame_equal(df.lazy().select(pds.query_lcm("a", other).cast(pl.Int64)).collect(), res)


@pytest.mark.parametrize(
    "df, x, res",
    [
        (
            pl.DataFrame({"a": [1, 2, 3]}),
            0.1,
            pl.DataFrame({"a": [0.4]}),
        ),
    ],
)
def test_trapz(df, x, res):
    assert_frame_equal(df.select(pds.integrate_trapz("a", x=x)), res)


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame(
                {"y": [1, 0, 1, 1, 1, 0, 0, 1], "a": ["a", "b", "c", "a", "b", "c", "a", "a"]}
            ),
            pl.DataFrame({"y": [0.6277411625893767]}),
        ),
        (
            pl.DataFrame({"y": [1] * 8, "a": ["a", "b", "c", "a", "b", "c", "a", "a"]}),
            pl.DataFrame({"y": [-0.0]}),
        ),
    ],
)
def test_cond_entropy(df, res):
    assert_frame_equal(df.select(pds.query_cond_entropy("y", "a")), res)

    assert_frame_equal(df.lazy().select(pds.query_cond_entropy("y", "a")).collect(), res)


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame(
                {
                    "y": [0, 1, 2, 0, 1],
                    "pred": [
                        [0.1, 0.5, 0.4],
                        [0.2, 0.6, 0.2],
                        [0.4, 0.1, 0.5],
                        [0.9, 0.05, 0.05],
                        [0.2, 0.5, 0.3],
                    ],
                }
            ),
            pl.DataFrame({"a": [0.8610131187075506]}),
        ),
    ],
)
def test_cross_entropy(df, res):
    assert_frame_equal(df.select(pds.query_cat_cross_entropy("y", "pred").alias("a")), res)

    assert_frame_equal(
        df.lazy().select(pds.query_cat_cross_entropy("y", "pred").alias("a")).collect(),
        res,
    )


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": [[1, 2, 3], [2, 3]], "b": [[1, 3], [1]]}),
            pl.DataFrame({"res": [2 / 3, 0.0]}),
        ),
        (
            pl.DataFrame({"a": [["a", "b", "c"], ["b", "c"]], "b": [["a", "b"], ["c"]]}),
            pl.DataFrame({"res": [2 / 3, 0.5]}),
        ),
    ],
)
def test_jaccard_row(df, res):
    assert_frame_equal(df.select(pds.query_jaccard_row("a", "b").alias("res")), res)

    assert_frame_equal(
        df.lazy().select(pds.query_jaccard_row("a", "b").alias("res")).collect(), res
    )


# Hard to write generic tests because ncols can vary in X
def test_lstsq():
    df = pl.DataFrame({"y": [1, 2, 3, 4, 5], "a": [2, 3, 4, 5, 6], "b": [-1, -1, -1, -1, -1]})
    res = pl.DataFrame({"y": [[1.0, 1.0]]})
    assert_frame_equal(
        df.select(pds.query_lstsq(pl.col("a"), pl.col("b"), target="y", add_bias=False)), res
    )

    df = pl.DataFrame(
        {
            "y": [1, 2, 3, 4, 5],
            "a": [2, 3, 4, 5, 6],
        }
    )
    res = pl.DataFrame({"y": [[1.0, -1.0]]})
    assert_frame_equal(df.select(pds.query_lstsq(pl.col("a"), target="y", add_bias=True)), res)


# Hard to write generic tests because ncols can vary in X
def test_lstsq_skip_null():
    df = pl.DataFrame(
        {"y": [None, 9.5, 10.5, 11.5, 12.5], "a": [1, 9, 10, 11, 12], "b": [1, 0.5, 0.5, 0.5, 0.5]}
    )
    res = pl.DataFrame(
        {
            "pred": [float("nan"), 9.5, 10.5, 11.5, 12.5],
            "resid": [float("nan"), 0.0, 0.0, 0.0, 0.0],
        }
    )
    assert_frame_equal(
        df.select(
            pds.query_lstsq(
                pl.col("a"), pl.col("b"), target="y", skip_null=True, return_pred=True
            ).alias("result")
        ).unnest("result"),
        res,
    )


def test_lstsq_in_group_by():
    df = pl.DataFrame(
        {
            "A": [1] * 4 + [2] * 4,
            "Y": [1] * 8,
            "X1": [1, 2, 3, 4, 5, 6, 7, 8],
            "X2": [2, 3, 4, 1, 6, 7, 8, 5],
        }
    )

    first = df.filter(pl.col("A").eq(1)).with_columns(
        pds.query_lstsq(
            pl.col("X1"), pl.col("X2"), target=pl.col("Y"), add_bias=False, return_pred=True
        ).alias("pred")
    )

    second = df.filter(pl.col("A").eq(2)).with_columns(
        pds.query_lstsq(
            pl.col("X1"), pl.col("X2"), target=pl.col("Y"), add_bias=False, return_pred=True
        ).alias("pred")
    )

    test = (
        df.group_by("A", maintain_order=True)
        .agg(
            "Y",
            "X1",
            "X2",
            pds.query_lstsq(
                pl.col("X1"), pl.col("X2"), target=pl.col("Y"), add_bias=False, return_pred=True
            ).alias("pred"),
        )
        .explode("Y", "X1", "X2", "pred")
    )

    test_first = test.filter(pl.col("A") == 1)
    test_second = test.filter(pl.col("A") == 2)

    assert_frame_equal(first, test_first)
    assert_frame_equal(second, test_second)


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]}),
            pl.DataFrame({"j": [2 / 3]}),
        ),
    ],
)
def test_jaccard_col(df, res):
    assert_frame_equal(df.select(pds.query_jaccard_col("a", "b").alias("j")), res)

    assert_frame_equal(
        df.lazy().select(df.select(pds.query_jaccard_col("a", "b").alias("j"))).collect(), res
    )


@pytest.mark.parametrize(
    "df, dtype, join_by, res",
    [
        (
            pl.DataFrame(
                {
                    "a": [
                        "0% of my time",
                        "1% to 25% of my time",
                        "75% to 99% of my time",
                        "50% to 74% of my time",
                        "75% to 99% of my time",
                        "50% to 74% of my time",
                    ]
                }
            ),
            pl.Int64,
            "",
            pl.DataFrame({"a": [[0], [1, 25], [75, 99], [50, 74], [75, 99], [50, 74]]}),
        ),
        (
            pl.DataFrame(
                {
                    "a": [
                        "0% of my time",
                        "1% to 25% of my time",
                        "75% to 99% of my time",
                        "50% to 74% of my time",
                        "75% to 99% of my time",
                        "50% to 74% of my time",
                    ]
                }
            ),
            pl.Utf8,
            "-",
            pl.DataFrame({"a": ["0", "1-25", "75-99", "50-74", "75-99", "50-74"]}),
        ),
    ],
)
def test_extract_numbers(df, dtype, join_by, res):
    assert_frame_equal(df.select(pds.extract_numbers("a", join_by=join_by, dtype=dtype)), res)


@pytest.mark.parametrize(
    "df, threshold, res",
    [
        (
            pl.DataFrame({"a": pl.Series([1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0])}),
            0,
            8,
        ),
        (
            pl.DataFrame(
                {
                    "a": pl.Series(
                        [
                            1,
                            0,
                            0,
                            1,
                            1,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                        ]
                    )
                }
            ),
            0,
            9,
        ),
        (
            pl.DataFrame(
                {
                    "a": pl.Series(
                        [
                            1,
                            0,
                            0,
                            1,
                            1,
                            1,
                            1,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                            0,
                            1,
                            0,
                        ]
                    )
                }
            ),
            0,
            10,
        ),
    ],
)
def test_lempel_ziv_complexity(df, threshold, res):
    test = df.select(pds.query_lempel_ziv(pl.col("a") > threshold, as_ratio=False))
    assert test.item(0, 0) == res


def test_ks_stats():
    from scipy.stats import ks_2samp
    import numpy as np

    a = np.random.random(size=1000)
    b = np.random.random(size=1000)
    df = pl.DataFrame({"a": a, "b": b})

    stats = ks_2samp(a, b).statistic
    # Only statistic for now
    res = df.select(pds.query_ks_2samp("a", "b").struct.field("statistic")).item(0, 0)

    assert np.isclose(stats, res)


@pytest.mark.parametrize(
    "df, k, dist, res",
    [
        (
            pl.DataFrame(dict(x=[1, 2, 10], y=[2, 5, 10])),
            1,
            "l2",
            5.675998737756144,
        ),
    ],
)
def test_knn_entropy(df, k, dist, res):
    test = df.select(pds.query_knn_entropy("x", "y", k=k, dist=dist)).item(0, 0)

    assert np.isclose(test, res)

    test_par = df.select(pds.query_knn_entropy("x", "y", k=k, dist=dist, parallel=True)).item(0, 0)

    assert np.isclose(test_par, res)


@pytest.mark.parametrize(
    "df, eq_var",
    [
        (
            pl.DataFrame({"a": np.random.normal(size=1_000), "b": np.random.normal(size=1_000)}),
            True,
        ),
        (
            pl.DataFrame({"a": np.random.normal(size=1_000), "b": np.random.normal(size=1_000)}),
            False,
        ),
    ],
)
def test_ttest_ind(df, eq_var):
    from scipy.stats import ttest_ind

    res = df.select(pds.query_ttest_ind("a", "b", equal_var=eq_var))
    res = res.item(0, 0)  # A dictionary
    statistic = res["statistic"]
    pvalue = res["pvalue"]

    scipy_res = ttest_ind(df["a"].to_numpy(), df["b"].to_numpy(), equal_var=eq_var)

    assert np.isclose(statistic, scipy_res.statistic)
    assert np.isclose(pvalue, scipy_res.pvalue)


@pytest.mark.parametrize(
    "df",
    [
        (
            pl.DataFrame(
                {
                    "a": np.random.normal(size=1_000),
                    "b": list(np.random.normal(size=998)) + [None, None],
                }
            )
        ),
    ],
)
def test_welch_t(df):
    from scipy.stats import ttest_ind

    res = df.select(pds.query_ttest_ind("a", "b", equal_var=False))
    res = res.item(0, 0)  # A dictionary
    statistic = res["statistic"]
    pvalue = res["pvalue"]

    s1 = df["a"].drop_nulls().to_numpy()
    s2 = df["b"].drop_nulls().to_numpy()
    scipy_res = ttest_ind(s1, s2, equal_var=False)

    assert np.isclose(statistic, scipy_res.statistic)
    assert np.isclose(pvalue, scipy_res.pvalue)


@pytest.mark.parametrize(
    "df",
    [
        (
            pl.DataFrame(
                {
                    "x": ["a"] * 200 + ["b"] * 300 + ["c"] * 500 + ["d"] * 200,
                    "y": [1] * 800 + [2] * 400,
                }
            )
        ),
    ],
)
def test_chi2(df):
    import pandas as pd
    from scipy.stats import chi2_contingency

    res = df.select(pds.query_chi2("x", "y")).item(0, 0)
    stats, p = res["statistic"], res["pvalue"]

    df2 = df.to_pandas()
    contigency = pd.crosstab(index=df2["x"], columns=df2["y"])
    sp_res = chi2_contingency(contigency.to_numpy(), correction=True)
    sp_stats, sp_p = sp_res.statistic, sp_res.pvalue
    assert np.isclose(stats, sp_stats)
    assert np.isclose(p, sp_p)


@pytest.mark.parametrize(
    "df",
    [
        (pl.DataFrame({"a": np.random.random(size=100)})),
    ],
)
def test_normal_test(df):
    from scipy.stats import normaltest

    res = df.select(pds.normal_test("a"))
    res = res.item(0, 0)  # A dictionary
    statistic = res["statistic"]
    pvalue = res["pvalue"]

    scipy_res = normaltest(df["a"].to_numpy())

    assert np.isclose(statistic, scipy_res.statistic)
    assert np.isclose(pvalue, scipy_res.pvalue)


@pytest.mark.parametrize(
    "df",
    [
        (pl.DataFrame({"a": 1000 * np.random.random(size=100)})),
    ],
)
def test_expit(df):
    from scipy.special import expit

    res = df.select(pds.expit("a"))["a"].to_numpy()
    scipy_res = expit(df["a"].to_numpy())
    assert np.isclose(res, scipy_res, equal_nan=True).all()


@pytest.mark.parametrize(
    "df",
    [
        (pl.DataFrame({"a": [0.0, 1.0, 2.0] + list(np.random.random(size=100))})),
    ],
)
def test_logit(df):
    from scipy.special import logit

    res = df.select(pds.logit("a"))["a"].to_numpy()
    scipy_res = logit(df["a"].to_numpy())
    assert np.isclose(res, scipy_res, equal_nan=True).all()


@pytest.mark.parametrize(
    "df",
    [
        (pl.DataFrame({"a": [0.0] + list(100 * np.random.random(size=100))})),
    ],
)
def test_gamma(df):
    from scipy.special import gamma

    res = df.select(pds.gamma("a"))["a"].to_numpy()
    scipy_res = gamma(df["a"].to_numpy())
    assert np.isclose(res, scipy_res, equal_nan=True).all()


def test_multiclass_roc_auc():
    from sklearn.metrics import roc_auc_score

    def roc_auc_random_data(size: int = 2000) -> pl.DataFrame:
        df = pl.DataFrame(
            {
                "id": range(size),
            }
        ).with_columns(
            pl.col("id").cast(pl.UInt64),
            pl.col("id").stats.rand_uniform(low=0.0, high=1.0).alias("val1"),
            pl.col("id").stats.rand_uniform(low=0.0, high=1.0).alias("val2"),
            pl.col("id").stats.rand_uniform(low=0.0, high=1.0).alias("val3"),
            pl.col("id").mod(3).alias("actuals"),
        )
        # Need to normalize to make sure this is valid ROC AUC data
        return (
            df.lazy()
            .with_columns(
                pl.concat_list(
                    pl.col("val1")
                    / pl.sum_horizontal(pl.col("val1"), pl.col("val2"), pl.col("val3")),
                    pl.col("val2")
                    / pl.sum_horizontal(pl.col("val1"), pl.col("val2"), pl.col("val3")),
                    pl.col("val3")
                    / pl.sum_horizontal(pl.col("val1"), pl.col("val2"), pl.col("val3")),
                ).alias("pred")
            )
            .select(
                pl.col("actuals"),
                pl.col("pred"),
            )
            .collect()
        )

    df = roc_auc_random_data()
    y_pred = np.stack(df["pred"].to_numpy())
    y_true = df["actuals"]

    macro = df.select(pds.query_multi_roc_auc("actuals", "pred", 3, "macro")).item(0, 0)
    macro_sklearn = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")

    weighted = df.select(pds.query_multi_roc_auc("actuals", "pred", 3, "weighted")).item(0, 0)
    weighted_sklearn = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")

    assert np.isclose(macro, macro_sklearn, rtol=1e-10, atol=1e-12)
    assert np.isclose(weighted, weighted_sklearn, rtol=1e-10, atol=1e-10)


def test_precision_recall_roc_auc():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    df = pl.DataFrame(
        {
            "a": np.random.random(size=5_000),
            "b": np.random.random(size=5_000),
            "y": np.round(np.random.random(size=5_000)).astype(int),
        }
    )
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        res = df.select(
            pds.query_binary_metrics("y", "a", threshold=threshold).alias("metrics")
        ).unnest("metrics")
        precision_res = res.get_column("precision")[0]
        recall_res = res.get_column("recall")[0]
        roc_auc_res = res.get_column("roc_auc")[0]

        # precision, recall by hand
        predicted_prob = np.array(df["a"])
        predicted = predicted_prob >= threshold  # boolean
        actual = np.array(df["y"])  # .to_numpy()
        precision = actual[predicted].sum() / np.sum(predicted)
        recall = ((actual == 1) & (predicted == 1)).sum() / (actual.sum())

        assert np.isclose(precision, precision_res)
        assert np.isclose(recall, recall_res)
        assert np.isclose(roc_auc_score(actual, predicted_prob), roc_auc_res)


@pytest.mark.parametrize(
    "df, dist, k, res",
    [
        (
            pl.DataFrame({"id": range(5), "val1": range(5), "val2": range(5), "val3": range(5)}),
            "l2",
            2,
            # Remember that this counts self as well.
            pl.DataFrame({"nn": [[0, 1, 2], [1, 2, 0], [2, 1, 3], [3, 2, 4], [4, 3, 2]]}),
        ),
    ],
)
def test_knn_ptwise(df, dist, k, res):
    df2 = df.select(
        pds.query_knn_ptwise(
            pl.col("val1"), pl.col("val2"), pl.col("val3"), index="id", dist=dist, k=k
        )
        .list.eval(pl.element().sort().cast(pl.UInt32))
        .alias("nn")
    )
    res = res.select(pl.col("nn").list.eval(pl.element().sort().cast(pl.UInt32)))
    assert_frame_equal(df2, res)


def test_knn_ptwise_skip():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "a1": [0.1, 0.2, 0.3, 0.4],
            "a2": [0.1, 0.2, 0.3, 0.4],
            "a3": [0.1, 0.2, 0.3, 0.4],
            "can_eval": [1, 0, 1, 1],
        }
    )

    res1 = df.with_columns(
        pds.query_knn_ptwise(
            "a1",
            "a2",
            "a3",
            index="id",
            k=1,
            dist="l2",  # squared l2
            parallel=False,
            eval_mask=pl.col("can_eval") == 1,
        ).alias("best friends")
    )
    friends = list(res1["best friends"])
    assert friends[1] is None  # should be true because we are skipping this evaluation
    assert list(friends[0]) == [0, 1]  # id 1 can still be a neighbor

    res2 = df.with_columns(
        pds.query_knn_ptwise(
            "a1",
            "a2",
            "a3",
            index="id",
            k=1,
            dist="l2",  # squared l2
            parallel=False,
            data_mask=pl.col("can_eval") == 1,
        ).alias("best friends")
    )
    friends = list(res2["best friends"])
    assert list(friends[0]) == [0, 2]  # id 1 cannot be a neighbor anymore
    all_neighbors = []
    for f in friends:
        all_neighbors += list(f)
    assert 1 not in set(all_neighbors)  # id 1 cannot be a neighbor anymore

    res3 = df.with_columns(
        pds.query_knn_ptwise(
            "a1",
            "a2",
            "a3",
            index="id",
            k=1,
            dist="l2",  # squared l2
            parallel=False,
            eval_mask=pl.col("can_eval") == 1,
            data_mask=pl.col("can_eval") == 1,
        ).alias("best friends")
    )
    friends = list(res3["best friends"])
    # Now 1 cannot be a neighbor, and we don't evaluate for 1
    assert friends[1] is None  # should be true because we are skipping this evaluation
    assert list(friends[0]) == [0, 2]  # id 1 can still be a neighbor


@pytest.mark.parametrize(
    "df, x, dist, k, res",
    [
        (  # Only the first row is the nearest neighbor to [0.5, 0.5, 0.5]
            pl.DataFrame(
                {"id": [1, 2], "val1": [0.1, 0.2], "val2": [0.1, 0.3], "val3": [0.1, 0.4]}
            ),
            [0.5, 0.5, 0.5],
            "cosine",
            1,
            pl.DataFrame({"id": [1]}),
        ),
    ],
)
def test_knn_filter(df, x, dist, k, res):
    test = df.filter(
        pds.query_knn_filter(pl.col("val1"), pl.col("val2"), pl.col("val3"), pt=x, dist=dist, k=k)
    ).select(pl.col("id"))

    assert_frame_equal(test, res)


@pytest.mark.parametrize(
    "df, r, dist, res",
    [
        (
            pl.DataFrame({"x": range(5), "y": range(5), "z": range(5)}),
            4,
            "l2",
            pl.DataFrame({"nb_cnt": [2, 3, 3, 3, 2]}),  # A point is always its own neighbor
        ),
    ],
)
def test_nb_cnt(df, r, dist, res):
    test = df.select(
        pds.query_nb_cnt(
            r,
            pl.col("x"),
            pl.col("y"),
            pl.col("z"),  # Columns used as the coordinates in n-d space
            dist=dist,
        )
        .cast(pl.UInt32)
        .alias("nb_cnt")
    )
    assert_frame_equal(test, res.with_columns(pl.col("nb_cnt").cast(pl.UInt32)))


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame(
                {
                    "x1": [51.5007],
                    "x2": [0.1246],
                    "y1": [40.6892],
                    "y2": [74.0445],
                }
            ),
            pl.DataFrame({"dist": [5574.840456848555]}),
        ),
    ],
)
def test_haversine(df, res):
    test = df.select(
        pds.haversine(pl.col("x1"), pl.col("x2"), pl.col("y1"), pl.col("y2")).alias("dist")
    )
    assert_frame_equal(test, res)


@pytest.mark.parametrize(
    "s, res",
    [
        (list(range(100)), 0.010471299867295437),
        (np.sin(2 * np.pi * np.arange(3000) / 100), 0.16367903754688098),
    ],
)
def test_sample_entropy(s, res):
    # Test 1's answer comes from comparing result with Tsfresh
    # Thest 2's answer comes from running this using the Python code on Wikipedia
    df = pl.Series(name="a", values=s).to_frame()
    entropy = df.select(pds.query_sample_entropy(pl.col("a"))).item(0, 0)
    assert np.isclose(entropy, res, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    "s, m, r, scale, res",
    [
        ([1], 2, 0.5, False, float("nan")),
        ([12, 13, 15, 16, 17] * 10, 2, 0.9, True, 0.282456191276673),
        ([1.4, -1.3, 1.7, -1.2], 2, 0.5, False, 0.0566330122651324),
        (
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            2,
            0.5,
            False,
            0.002223871246127107,
        ),
        (
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
            2,
            0.5,
            False,
            0.47133806162842484,
        ),
        ([85, 80, 89] * 17, 2, 3, False, 1.099654110658932e-05),
        ([85, 80, 89] * 17, 2, 3, True, 0.0),
    ],
)
def test_apprximate_entropy(s, m, r, scale, res):
    df = pl.Series(name="a", values=s).to_frame()
    entropy = df.select(
        pds.query_approx_entropy("a", m=m, filtering_level=r, scale_by_std=scale)
    ).item(0, 0)
    assert np.isclose(entropy, res, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    "df, n_bins, res",
    [
        (  # 4 * (0.1 - 0.0001) * np.log(0.1 / 0.0001) + (0.1 - 0.5) * np.log(0.1 / 0.5)
            pl.DataFrame({"ref": range(1000), "act": list(range(500)) + [600] * 500}),
            10,
            3.4041141744549024,
        ),
    ],
)
def test_psi(df, n_bins, res):
    ans = df.select(pds.query_psi("act", pl.col("ref"), n_bins=n_bins)).item(0, 0)
    assert np.isclose(ans, res)


@pytest.mark.parametrize(
    "df, res",
    [
        (  # discrete version of the above example
            pl.DataFrame(
                {
                    "ref": [0] * 100
                    + [1] * 100
                    + [2] * 100
                    + [3] * 100
                    + [4] * 100
                    + [5] * 100
                    + [6] * 100
                    + [7] * 100
                    + [8] * 100
                    + [9] * 100,
                    "act": [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [6] * 500,
                }
            ),
            3.4041141744549024,
        ),
    ],
)
def test_psi_discrete(df, res):
    ans = df.select(pds.query_psi_discrete("act", pl.col("ref"))).item(0, 0)
    assert np.isclose(ans, res)


@pytest.mark.parametrize(
    "df, target, path, cost",
    [
        # It is easier to provide a frame that needs to be exploded by
        (
            pl.DataFrame(
                {
                    "id": range(5),
                    "conn": [[1, 2, 3, 4], [2, 3], [4], [0, 1, 2], [1]],
                    "cost": [[0.4, 0.3, 0.2, 0.1], [0.1, 1.0], [0.5], [0.1, 0.1, 0.1], [0.1]],
                }
            ).with_columns(
                pl.col("id").cast(pl.UInt32), pl.col("conn").list.eval(pl.element().cast(pl.UInt32))
            ),
            1,
            [[4, 1], [], [4, 1], [1], [1]],
            [0.2, 0.0, 0.6, 0.1, 0.1],
        ),
    ],
)
def test_shortest_dist(df, target, path, cost):
    df = df.explode([pl.col("conn"), pl.col("cost")])

    res = (
        df.select(pds.query_shortest_path("id", "conn", target=target, cost="cost").alias("path"))
        .unnest("path")
        .sort("id")
    )

    for p, ans in zip(res["path"], path):
        assert list(p) == ans

    for c, ans in zip(res["cost"], cost):
        assert c == ans
