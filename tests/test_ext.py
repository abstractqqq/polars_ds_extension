import pytest
import polars as pl
import math
import numpy as np
import polars_ds  # noqa: F401
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    "df",
    [
        (pl.DataFrame({"a": np.random.random(size=100)})),
        (pl.DataFrame({"a": np.random.normal(size=100)})),
    ],
)
def test_normal_test(df):
    from scipy.stats import normaltest

    res = df.select(pl.col("a").stats_ext.normal_test())
    res = res.item(0, 0)  # A dictionary
    statistic = res["statistic"]
    pvalue = res["pvalue"]

    scipy_res = normaltest(df["a"].to_numpy())

    assert np.isclose(statistic, scipy_res.statistic)
    assert np.isclose(pvalue, scipy_res.pvalue)


@pytest.mark.parametrize(
    "df",
    [
        (pl.DataFrame({"a": np.random.normal(size=1_000), "b": np.random.normal(size=1_000)})),
    ],
)
def test_ttest_ind(df):
    from scipy.stats import ttest_ind

    res = df.select(pl.col("a").stats_ext.ttest_ind(pl.col("b"), equal_var=True))
    res = res.item(0, 0)  # A dictionary
    statistic = res["statistic"]
    pvalue = res["pvalue"]

    scipy_res = ttest_ind(df["a"].to_numpy(), df["b"].to_numpy(), equal_var=True)

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

    res = df.select(pl.col("a").stats_ext.ttest_ind(pl.col("b"), equal_var=False))
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
                    "target": np.random.randint(0, 3, size=1_000),
                    "a": np.random.normal(size=1_000),
                }
            )
        ),
    ],
)
def test_f_test(df):
    from sklearn.feature_selection import f_classif

    res = df.select(pl.col("target").stats_ext.f_test(pl.col("a")))
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
    assert_frame_equal(df.select(pl.col("a").num_ext.gcd(other)), res)

    assert_frame_equal(df.lazy().select(pl.col("a").num_ext.gcd(other)).collect(), res)


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
    assert_frame_equal(df.select(pl.col("a").num_ext.lcm(other)), res)

    assert_frame_equal(df.lazy().select(pl.col("a").num_ext.lcm(other)).collect(), res)


@pytest.mark.parametrize(
    "df, p",
    [
        (
            pl.DataFrame(
                {
                    "a": [0.1 + x / 1000 for x in range(1000)],
                    "b": pl.Series(range(1000), dtype=pl.Int32),
                }
            ),
            pl.col("b"),
        ),
        (
            pl.DataFrame(
                {
                    "a": [0.1 + x / 1000 for x in range(1000)],
                    "b": pl.Series(range(1000), dtype=pl.Int32),
                }
            ),
            10,
        ),
        (
            pl.DataFrame(
                {
                    "a": [math.inf, math.nan],
                }
            ),
            2,
        ),
    ],
)
def test_powi(df, p):
    # The reason I avoided 0 is that
    # In polars 0^0 = 1, which is wrong.
    # In polars-ds, this will be mapped to NaN.
    assert_frame_equal(df.select(pl.col("a").num_ext.powi(p)), df.select(pl.col("a").pow(p)))


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
    assert_frame_equal(df.select(pl.col("a").num_ext.trapz(x=x)), res)


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
    assert_frame_equal(df.select(pl.col("y").num_ext.cond_entropy(pl.col("a"))), res)

    assert_frame_equal(
        df.lazy().select(pl.col("y").num_ext.cond_entropy(pl.col("a"))).collect(), res
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
def test_list_jaccard(df, res):
    assert_frame_equal(df.select(pl.col("a").num_ext.list_jaccard(pl.col("b")).alias("res")), res)

    assert_frame_equal(
        df.lazy().select(pl.col("a").num_ext.list_jaccard(pl.col("b")).alias("res")).collect(), res
    )


# Hard to write generic tests because ncols can vary in X
def test_lstsq():
    df = pl.DataFrame({"y": [1, 2, 3, 4, 5], "a": [2, 3, 4, 5, 6], "b": [-1, -1, -1, -1, -1]})
    res = pl.DataFrame({"y": [[1.0, 1.0]]})
    assert_frame_equal(
        df.select(pl.col("y").num_ext.lstsq(pl.col("a"), pl.col("b"), add_bias=False)), res
    )

    df = pl.DataFrame(
        {
            "y": [1, 2, 3, 4, 5],
            "a": [2, 3, 4, 5, 6],
        }
    )
    res = pl.DataFrame({"y": [[1.0, -1.0]]})
    assert_frame_equal(df.select(pl.col("y").num_ext.lstsq(pl.col("a"), add_bias=True)), res)


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
    assert_frame_equal(df.select(pl.col("a").num_ext.jaccard(pl.col("b")).alias("j")), res)

    assert_frame_equal(
        df.lazy().select(pl.col("a").num_ext.jaccard(pl.col("b")).alias("j")).collect(), res
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
    assert_frame_equal(df.select(pl.col("a").str_ext.snowball()), res)

    assert_frame_equal(df.select(pl.col("a").str_ext.snowball(parallel=True)), res)

    assert_frame_equal(df.lazy().select(pl.col("a").str_ext.snowball()).collect(), res)


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
    assert_frame_equal(df.select(pl.col("a").str_ext.hamming(pl.col("b"))), res)
    assert_frame_equal(df.select(pl.col("a").str_ext.hamming(pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pl.col("a").str_ext.hamming(pl.col("b"))).collect(), res)


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
    assert_frame_equal(df.select(pl.col("a").str_ext.jaro(pl.col("b"))), res)
    assert_frame_equal(df.select(pl.col("a").str_ext.jaro(pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pl.col("a").str_ext.jaro(pl.col("b"))).collect(), res)


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
        pl.col("a").str_ext.ac_match(
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
    assert_frame_equal(
        df.select(pl.col("a").str_ext.ac_replace(patterns=pat, replacements=repl)), res
    )

    assert_frame_equal(
        df.lazy().select(pl.col("a").str_ext.ac_replace(patterns=pat, replacements=repl)).collect(),
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
    assert_frame_equal(df.select(pl.col("a").str_ext.levenshtein(pl.col("b"))), res)

    assert_frame_equal(df.select(pl.col("a").str_ext.levenshtein(pl.col("b"), parallel=True)), res)

    assert_frame_equal(
        df.lazy().select(pl.col("a").str_ext.levenshtein(pl.col("b"))).collect(), res
    )


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
        df.select(pl.col("a").str_ext.levenshtein_within(pl.col("b"), bound=bound)), res
    )

    assert_frame_equal(
        df.select(pl.col("a").str_ext.levenshtein_within(pl.col("b"), bound=bound, parallel=True)),
        res,
    )

    assert_frame_equal(
        df.lazy()
        .select(pl.col("a").str_ext.levenshtein_within(pl.col("b"), bound=bound))
        .collect(),
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
    assert_frame_equal(df.select(pl.col("a").str_ext.osa(pl.col("b"))), res)

    assert_frame_equal(df.select(pl.col("a").str_ext.osa(pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pl.col("a").str_ext.osa(pl.col("b"))).collect(), res)


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
    assert_frame_equal(df.select(pl.col("a").str_ext.sorensen_dice(pl.col("b"))), res)

    assert_frame_equal(
        df.select(pl.col("a").str_ext.sorensen_dice(pl.col("b"), parallel=True)), res
    )

    assert_frame_equal(
        df.lazy().select(pl.col("a").str_ext.sorensen_dice(pl.col("b"))).collect(), res
    )


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
    assert_frame_equal(
        df.select(pl.col("a").str_ext.str_jaccard(pl.col("b"), substr_size=size)), res
    )
    assert_frame_equal(
        df.select(pl.col("a").str_ext.str_jaccard(pl.col("b"), substr_size=size, parallel=True)),
        res,
    )
    assert_frame_equal(
        df.lazy()
        .select(pl.col("a").str_ext.str_jaccard(pl.col("b"), substr_size=size, parallel=True))
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
    ans = df.select(pl.col("a").str_ext.freq_removal(lower=lower, upper=upper).list.sort())
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
        df.select(pl.col("a").str_ext.extract_numbers(join_by=join_by, dtype=dtype)), res
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
        df.select(pl.col("a").str_ext.merge_infreq(min_count=min_count, min_frac=min_frac)), res
    )


def test_ks_stats():
    from scipy.stats import ks_2samp
    import numpy as np

    a = np.random.random(size=1000)
    b = np.random.random(size=1000)
    df = pl.DataFrame({"a": a, "b": b})

    stats = ks_2samp(a, b).statistic
    # Only statistic for now
    res = df.select(pl.col("a").stats_ext.ks_stats(pl.col("b"))).item(0, 0)

    assert np.isclose(stats, res)


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
            pl.col("y")
            .num_ext.binary_metrics_combo(pl.col("a"), threshold=threshold)
            .alias("metrics")
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
