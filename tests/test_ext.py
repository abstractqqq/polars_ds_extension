import pytest
import polars as pl
import numpy as np
import polars_ds as pld
from polars.testing import assert_frame_equal


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

    res = df.select(pl.col("target").stats.f_test(pl.col("a")))
    res = res.item(0, 0)  # A dictionary
    statistic = res["statistic"]
    pvalue = res["pvalue"]

    scikit_res = f_classif(df["a"].to_numpy().reshape(-1, 1), df["target"].to_numpy())
    scikit_s = scikit_res[0][0]
    scikit_p = scikit_res[1][0]

    assert np.isclose(statistic, scikit_s)
    assert np.isclose(pvalue, scikit_p)


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
    assert_frame_equal(df.select(pl.col("a").num.gcd(other)), res)

    assert_frame_equal(df.lazy().select(pl.col("a").num.gcd(other)).collect(), res)


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
    assert_frame_equal(df.select(pl.col("a").num.lcm(other)), res)

    assert_frame_equal(df.lazy().select(pl.col("a").num.lcm(other)).collect(), res)


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
    assert_frame_equal(df.select(pl.col("a").num.trapz(x=x)), res)


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
    assert_frame_equal(df.select(pl.col("y").num.cond_entropy(pl.col("a"))), res)

    assert_frame_equal(df.lazy().select(pl.col("y").num.cond_entropy(pl.col("a"))).collect(), res)


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
def test_list_jaccard(df, res):
    assert_frame_equal(df.select(pl.col("a").num.list_jaccard(pl.col("b")).alias("res")), res)

    assert_frame_equal(
        df.lazy().select(pl.col("a").num.list_jaccard(pl.col("b")).alias("res")).collect(), res
    )


# Hard to write generic tests because ncols can vary in X
def test_lstsq():
    df = pl.DataFrame({"y": [1, 2, 3, 4, 5], "a": [2, 3, 4, 5, 6], "b": [-1, -1, -1, -1, -1]})
    res = pl.DataFrame({"y": [[1.0, 1.0]]})
    assert_frame_equal(
        df.select(pl.col("y").num.lstsq(pl.col("a"), pl.col("b"), add_bias=False)), res
    )

    df = pl.DataFrame(
        {
            "y": [1, 2, 3, 4, 5],
            "a": [2, 3, 4, 5, 6],
        }
    )
    res = pl.DataFrame({"y": [[1.0, -1.0]]})
    assert_frame_equal(df.select(pl.col("y").num.lstsq(pl.col("a"), add_bias=True)), res)


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]}),
            pl.DataFrame({"j": [2 / 3]}),
        ),
    ],
)
def test_col_jaccard(df, res):
    assert_frame_equal(df.select(pl.col("a").num.jaccard(pl.col("b")).alias("j")), res)

    assert_frame_equal(
        df.lazy().select(pl.col("a").num.jaccard(pl.col("b")).alias("j")).collect(), res
    )


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": ["thanks", "thank", "thankful"]}),
            pl.DataFrame({"a": ["thank", "thank", "thank"]}),
        ),
        (
            pl.DataFrame({"a": ["playful", "playing", "play", "played", "plays"]}),
            pl.DataFrame({"a": ["play", "play", "play", "play", "play"]}),
        ),
    ],
)
def test_snowball(df, res):
    assert_frame_equal(df.select(pl.col("a").str2.snowball()), res)

    assert_frame_equal(df.select(pl.col("a").str2.snowball(parallel=True)), res)

    assert_frame_equal(df.lazy().select(pl.col("a").str2.snowball()).collect(), res)


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame(
                {
                    "a": ["karolin", "karolin", "kathrin", "0000", "2173896"],
                    "b": ["kathrin", "kerstin", "kerstin", "1111", "2233796"],
                }
            ),
            pl.DataFrame({"a": pl.Series([3, 3, 4, 4, 3], dtype=pl.UInt32)}),
        ),
    ],
)
def test_hamming(df, res):
    assert_frame_equal(df.select(pl.col("a").str2.hamming(pl.col("b"))), res)
    assert_frame_equal(df.select(pl.col("a").str2.hamming(pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pl.col("a").str2.hamming(pl.col("b"))).collect(), res)


@pytest.mark.parametrize(
    "df, res",
    [
        (  # From Wikipedia
            pl.DataFrame(
                {
                    "a": ["FAREMVIEL"],
                    "b": ["FARMVILLE"],
                }
            ),
            pl.DataFrame({"a": pl.Series([(1 / 3) * (16 / 9 + 7 / 8)], dtype=pl.Float64)}),
        ),
    ],
)
def test_jaro(df, res):
    assert_frame_equal(df.select(pl.col("a").str2.jaro(pl.col("b"))), res)
    assert_frame_equal(df.select(pl.col("a").str2.jaro(pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pl.col("a").str2.jaro(pl.col("b"))).collect(), res)


@pytest.mark.parametrize(
    "df, pat, res",
    [
        (  # From Wikipedia
            pl.DataFrame(
                {
                    "a": ["Nobody likes maple in their apple flavored Snapple."],
                }
            ),
            ["apple", "maple", "snapple"],
            [1, 0, 2],
        ),
    ],
)
def test_ac_match(df, pat, res):
    ans = df.select(
        pl.col("a").str2.ac_match(
            patterns=pat, case_sensitive=True, match_kind="standard", return_str=False
        )
    ).item(0, 0)
    ans = list(ans)

    assert ans == res

    ans_strs = [pat[i] for i in ans]
    res_strs = [pat[i] for i in res]

    assert ans_strs == res_strs


@pytest.mark.parametrize(
    "df, pat, repl, res",
    [
        (  # From Wikipedia
            pl.DataFrame(
                {
                    "a": ["hate 123 hate, poor 123, sad !23dc"],
                }
            ),
            ["hate", "poor", "sad"],
            ["love", "wealthy", "happy"],
            pl.DataFrame(
                {
                    "a": ["love 123 love, wealthy 123, happy !23dc"],
                }
            ),
        ),
    ],
)
def test_ac_replace(df, pat, repl, res):
    assert_frame_equal(df.select(pl.col("a").str2.ac_replace(patterns=pat, replacements=repl)), res)

    assert_frame_equal(
        df.lazy().select(pl.col("a").str2.ac_replace(patterns=pat, replacements=repl)).collect(),
        res,
    )


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": ["kitten", "mary", "may"], "b": ["sitting", "merry", "mayer"]}),
            pl.DataFrame({"a": pl.Series([3, 2, 2], dtype=pl.UInt32)}),
        ),
        (
            pl.DataFrame(
                {
                    "a": [
                        "Ostroróg",
                        "Hätönen",
                        "Kõivsaar",
                        "Pöitel",
                        "Vystrčil",
                        "Särki",
                        "Chreptavičienė",
                        "Väänänen",
                        "Führus",
                        "Könönen",
                        "Väänänen",
                        "Łaszczyński",
                        "Pärnselg",
                        "Könönen",
                        "Piątkowski",
                        "D’Amore",
                        "Körber",
                        "Särki",
                        "Kärson",
                        "Węgrzyn",
                    ],
                    "b": [
                        "Könönen",
                        "Hätönen",
                        "Wyżewski",
                        "Jäger",
                        "Hätönen",
                        "Mäns",
                        "Chreptavičienė",
                        "Väänänen",
                        "Ahısha",
                        "Jürist",
                        "Vainjärv",
                        "Łaszczyński",
                        "Pärnselg",
                        "Führus",
                        "Kübarsepp",
                        "Németi",
                        "Räheso",
                        "Käri",
                        "Jäger",
                        "Setälä",
                    ],
                }
            ),
            pl.DataFrame(
                {
                    "a": pl.Series(
                        [8, 0, 8, 5, 7, 4, 0, 0, 6, 7, 6, 0, 0, 7, 10, 6, 6, 2, 5, 7],
                        dtype=pl.UInt32,
                    )
                }
            ),
        ),
    ],
)
def test_levenshtein(df, res):
    assert_frame_equal(df.select(pl.col("a").str2.levenshtein(pl.col("b"))), res)

    assert_frame_equal(df.select(pl.col("a").str2.levenshtein(pl.col("b"), parallel=True)), res)

    assert_frame_equal(df.lazy().select(pl.col("a").str2.levenshtein(pl.col("b"))).collect(), res)


@pytest.mark.parametrize(
    "df, bound, res",
    [
        (
            pl.DataFrame(
                {"a": ["kitten", "mary", "may", None], "b": ["sitting", "merry", "mayer", ""]}
            ),
            2,
            pl.DataFrame({"a": pl.Series([False, True, True, None])}),
        ),
    ],
)
def test_levenshtein_within(df, bound, res):
    assert_frame_equal(
        df.select(pl.col("a").str2.levenshtein_within(pl.col("b"), bound=bound)), res
    )

    assert_frame_equal(
        df.select(pl.col("a").str2.levenshtein_within(pl.col("b"), bound=bound, parallel=True)),
        res,
    )

    assert_frame_equal(
        df.lazy().select(pl.col("a").str2.levenshtein_within(pl.col("b"), bound=bound)).collect(),
        res,
    )


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": ["CA", "AB", None], "b": ["ABC", "BA", "a"]}),
            pl.DataFrame({"a": pl.Series([3, 1, None], dtype=pl.UInt32)}),
        ),
    ],
)
def test_osa(df, res):
    assert_frame_equal(df.select(pl.col("a").str2.osa(pl.col("b"))), res)

    assert_frame_equal(df.select(pl.col("a").str2.osa(pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pl.col("a").str2.osa(pl.col("b"))).collect(), res)


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": ["kitten"], "b": ["sitting"]}),
            pl.DataFrame({"a": pl.Series([4 / 11], dtype=pl.Float64)}),
        ),
    ],
)
def test_sorensen_dice(df, res):
    assert_frame_equal(df.select(pl.col("a").str2.sorensen_dice(pl.col("b"))), res)

    assert_frame_equal(df.select(pl.col("a").str2.sorensen_dice(pl.col("b"), parallel=True)), res)

    assert_frame_equal(df.lazy().select(pl.col("a").str2.sorensen_dice(pl.col("b"))).collect(), res)


@pytest.mark.parametrize(
    "df, size, res",
    [
        (
            pl.DataFrame({"a": ["apple", "test", "moon"], "b": ["let", "tests", "sun"]}),
            2,
            pl.DataFrame({"a": pl.Series([0.2, 0.75, 0.0], dtype=pl.Float64)}),
        ),
        (
            pl.DataFrame({"a": ["apple", "test", "moon"], "b": ["let", "tests", "sun"]}),
            3,
            pl.DataFrame({"a": pl.Series([0.0, 2 / 3, 0.0], dtype=pl.Float64)}),
        ),
    ],
)
def test_str_jaccard(df, size, res):
    assert_frame_equal(df.select(pl.col("a").str2.str_jaccard(pl.col("b"), substr_size=size)), res)
    assert_frame_equal(
        df.select(pl.col("a").str2.str_jaccard(pl.col("b"), substr_size=size, parallel=True)),
        res,
    )
    assert_frame_equal(
        df.lazy()
        .select(pl.col("a").str2.str_jaccard(pl.col("b"), substr_size=size, parallel=True))
        .collect(),
        res,
    )


@pytest.mark.parametrize(
    "df, lower, upper, res",
    [
        (
            pl.DataFrame({"a": [["a", "b", "c"], ["a", "b"], ["a"]]}),
            0.05,
            0.6,
            pl.DataFrame({"a": [["b", "c"], ["b"], []]}),
            # 0.05 is count of 1, nothing has < 1 count. 0.6 is 2. "a" has > 2 count
            # so a is removed.
        ),
    ],
)
def test_freq_removal(df, lower, upper, res):
    ans = df.select(pl.col("a").str2.freq_removal(lower=lower, upper=upper).list.sort())
    assert_frame_equal(ans, res)


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
    assert_frame_equal(
        df.select(pl.col("a").str2.extract_numbers(join_by=join_by, dtype=dtype)), res
    )


@pytest.mark.parametrize(
    "df, min_count, min_frac, res",
    [
        (
            pl.DataFrame(
                {
                    "a": ["a", "b", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c"],
                }
            ),
            3,
            None,
            pl.DataFrame(
                {"a": ["a|b", "a|b", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c"]}
            ),
        ),
        (
            pl.DataFrame(
                {
                    "a": ["a", "b", "c", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d"],
                }
            ),
            None,
            0.1,
            pl.DataFrame(
                {
                    "a": [
                        "a|b|c",
                        "a|b|c",
                        "a|b|c",
                        "d",
                        "d",
                        "d",
                        "d",
                        "d",
                        "d",
                        "d",
                        "d",
                        "d",
                        "d",
                        "d",
                    ]
                }
            ),
        ),
    ],
)
def test_merge_infreq(df, min_count, min_frac, res):
    assert_frame_equal(
        df.select(pl.col("a").str2.merge_infreq(min_count=min_count, min_frac=min_frac)), res
    )


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
    test = df.select((pl.col("a") > threshold).num.lempel_ziv_complexity(as_ratio=False))
    assert test.item(0, 0) == res


def test_ks_stats():
    from scipy.stats import ks_2samp
    import numpy as np

    a = np.random.random(size=1000)
    b = np.random.random(size=1000)
    df = pl.DataFrame({"a": a, "b": b})

    stats = ks_2samp(a, b).statistic
    # Only statistic for now
    res = df.select(pl.col("a").stats.ks_stats(pl.col("b"))).item(0, 0)

    assert np.isclose(stats, res)


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

    res = df.select(pl.col("a").stats.ttest_ind(pl.col("b"), equal_var=eq_var))
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

    res = df.select(pl.col("a").stats.ttest_ind(pl.col("b"), equal_var=False))
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

    res = df.select(pl.col("x").stats.chi2(pl.col("y"))).item(0, 0)
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
        (pl.DataFrame({"a": np.random.normal(size=100)})),
    ],
)
def test_normal_test(df):
    from scipy.stats import normaltest

    res = df.select(pl.col("a").stats.normal_test())
    res = res.item(0, 0)  # A dictionary
    statistic = res["statistic"]
    pvalue = res["pvalue"]

    scipy_res = normaltest(df["a"].to_numpy())

    assert np.isclose(statistic, scipy_res.statistic)
    assert np.isclose(pvalue, scipy_res.pvalue)


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
            pl.col("y").num.binary_metrics_combo(pl.col("a"), threshold=threshold).alias("metrics")
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
        pl.col("id")
        .num.knn_ptwise(pl.col("val1"), pl.col("val2"), pl.col("val3"), dist=dist, k=k)
        .list.eval(pl.element().sort().cast(pl.UInt32))
        .alias("nn")
    )
    # Make sure the list inner types are both u64
    res = res.select(pl.col("nn").list.eval(pl.element().sort().cast(pl.UInt32)))

    assert_frame_equal(df2, res)


@pytest.mark.parametrize(
    "df, x, dist, k, res",
    [
        (
            pl.DataFrame({"id": range(5), "val1": range(5), "val2": range(5), "val3": range(5)}),
            [0.5, 0.5, 0.5],
            "l2",
            3,
            pl.DataFrame({"id": [0, 1, 2]}),
        ),
    ],
)
def test_knn_pt(df, x, dist, k, res):
    test = df.filter(
        pld.knn(x, pl.col("val1"), pl.col("val2"), pl.col("val3"), dist=dist, k=k)
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
        pld.query_nb_cnt(
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
        pld.haversine(pl.col("x1"), pl.col("x2"), pl.col("y1"), pl.col("y2")).alias("dist")
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
    entropy = df.select(pl.col("a").num.sample_entropy()).item(0, 0)
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
        pl.col("a").num.approximate_entropy(m=m, filtering_level=r, scale_by_std=scale)
    ).item(0, 0)
    assert np.isclose(entropy, res, atol=1e-12, equal_nan=True)
