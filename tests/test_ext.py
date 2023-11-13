import pytest
import polars as pl
import math
from polars_ds import NumExt, StrExt  # noqa: F401
from polars.testing import assert_frame_equal


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
def test_hamming_dist(df, res):
    assert_frame_equal(df.select(pl.col("a").str_ext.hamming_dist(pl.col("b"))), res)
    assert_frame_equal(df.select(pl.col("a").str_ext.hamming_dist(pl.col("b"), parallel=True)), res)
    assert_frame_equal(
        df.lazy().select(pl.col("a").str_ext.hamming_dist(pl.col("b"))).collect(), res
    )


@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({"a": ["kitten", "mary", "may"], "b": ["sitting", "merry", "mayer"]}),
            pl.DataFrame({"a": pl.Series([3, 2, 2], dtype=pl.UInt32)}),
        ),
    ],
)
def test_levenshtein_dist(df, res):
    assert_frame_equal(df.select(pl.col("a").str_ext.levenshtein_dist(pl.col("b"))), res)

    assert_frame_equal(
        df.select(pl.col("a").str_ext.levenshtein_dist(pl.col("b"), parallel=True)), res
    )
    assert_frame_equal(
        df.select(pl.col("a").str_ext.levenshtein_dist("may")),
        pl.DataFrame({"a": pl.Series([6, 1, 0], dtype=pl.UInt32)}),
    )
    assert_frame_equal(
        df.lazy().select(pl.col("a").str_ext.levenshtein_dist(pl.col("b"))).collect(), res
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
