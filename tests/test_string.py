from __future__ import annotations

import pytest
import polars as pl
import polars_ds as pds
from polars.testing import assert_frame_equal


def test_replace_non_ascii():
    df = pl.DataFrame({"x": ["mercy", "xbĤ", "ĤŇƏ"]})

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x")), pl.DataFrame({"x": ["mercy", "xb", ""]})
    )

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x", "?")),
        pl.DataFrame({"x": ["mercy", "xb?", "???"]}),
    )

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x", "??")),
        pl.DataFrame({"x": ["mercy", "xb??", "??????"]}),
    )


def test_remove_diacritics():
    df = pl.DataFrame({"x": ["mercy", "mèrcy", "françoise", "über"]})

    assert_frame_equal(
        df.select(pds.remove_diacritics("x")),
        pl.DataFrame({"x": ["mercy", "mercy", "francoise", "uber"]}),
    )


def test_normalize_string():
    df = pl.DataFrame({"x": ["\u0043\u0327"], "y": ["\u00c7"]}).with_columns(
        pl.col("x").eq(pl.col("y")).alias("is_equal"),
        pds.normalize_string("x", "NFC")
        .eq(pds.normalize_string("y", "NFC"))
        .alias("normalized_is_equal"),
    )

    assert df["is_equal"].sum() == 0
    assert df["normalized_is_equal"].sum() == df.height


def test_map_words():
    df = pl.DataFrame({"x": ["one two three", "onetwo three"]})

    assert_frame_equal(
        df.select(pds.map_words("x", {"two": "2"})),
        pl.DataFrame({"x": ["one 2 three", "onetwo three"]}),
    )

    assert_frame_equal(
        df.select(pds.map_words("x", {"two": "2", "three": "3"})),
        pl.DataFrame({"x": ["one 2 3", "onetwo 3"]}),
    )

    assert_frame_equal(
        df.select(pds.map_words("x", {"four": "4"})),
        pl.DataFrame({"x": ["one two three", "onetwo three"]}),
    )


def test_normalize_whitespace():
    df = pl.DataFrame({"x": ["a   b", "ab", "a b", "a\t\nb"]})

    assert_frame_equal(
        df.select(pds.normalize_whitespace("x")),
        pl.DataFrame({"x": ["a b", "ab", "a b", "a b"]}),
    )

    assert_frame_equal(
        df.select(pds.normalize_whitespace("x", only_spaces=True)),
        pl.DataFrame({"x": ["a b", "ab", "a b", "a\t\nb"]}),
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
    assert_frame_equal(df.select(pds.str_snowball("a")), res)

    assert_frame_equal(df.lazy().select(pds.str_snowball("a")).collect(), res)


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
    assert_frame_equal(df.select(pds.str_hamming("a", pl.col("b"))), res)
    assert_frame_equal(df.select(pds.str_hamming("a", pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pds.str_hamming("a", pl.col("b"))).collect(), res)


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
    assert_frame_equal(df.select(pds.str_jaro("a", pl.col("b"))), res)
    assert_frame_equal(df.select(pds.str_jaro("a", pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pds.str_jaro("a", pl.col("b"))).collect(), res)


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
    assert_frame_equal(df.select(pds.str_leven("a", pl.col("b"))), res)

    assert_frame_equal(df.lazy().select(pds.str_leven("a", pl.col("b"))).collect(), res)


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
def test_levenshtein_filter(df, bound, res):
    assert_frame_equal(
        df.select(pds.filter_by_levenshtein(pl.col("a"), pl.col("b"), bound=bound)), res
    )

    assert_frame_equal(
        df.select(pds.filter_by_levenshtein(pl.col("a"), pl.col("b"), bound=bound, parallel=True)),
        res,
    )

    assert_frame_equal(
        df.lazy()
        .select(pds.filter_by_levenshtein(pl.col("a"), pl.col("b"), bound=bound))
        .collect(),
        res,
    )


@pytest.mark.parametrize(
    "df, bound, res",
    [
        (
            pl.DataFrame(
                {
                    "a": ["AAAAA", "AAATT", "AATTT", "AAAAA", "AAAAA"],
                    "b": ["AAAAT", "AAAAA", "ATATA", "AAAAA", "TTTTT"],
                }
            ),
            2,
            pl.DataFrame({"a": pl.Series([True, True, False, True, False])}),
        ),
    ],
)
def test_hamming_filter(df, bound, res):
    assert_frame_equal(df.select(pds.filter_by_hamming("a", pl.col("b"), bound=bound)), res)

    assert_frame_equal(
        df.select(pds.filter_by_hamming("a", pl.col("b"), bound=bound, parallel=True)),
        res,
    )

    assert_frame_equal(
        df.lazy().select(pds.filter_by_hamming("a", pl.col("b"), bound=bound)).collect(),
        res,
    )


@pytest.mark.parametrize(
    "df, vocab, k, metric, res",
    [
        (
            pl.DataFrame(
                {
                    "a": ["AAAAA", "AAATT", "ATTTT", "AAAAA", "AAAAA"],
                }
            ),
            ["AAAAT", "AAAAA", "ATATA", "AAAAA", "TTTTT"],
            1,
            "hamming",
            pl.DataFrame({"a": pl.Series(["AAAAA", "AAAAT", "TTTTT", "AAAAA", "AAAAA"])}),
        ),
        (
            pl.DataFrame(
                {
                    "a": ["AAAAA", "AAATT", "ATTTT", "AAAAA", "AAAAA"],
                }
            ),
            ["AAAAT", "AAAAA", "ATATA", "AAAAA", "TTTTT"],
            2,
            "hamming",
            pl.DataFrame(
                {
                    "a": pl.Series(
                        [
                            ["AAAAA", "AAAAT"],
                            ["AAAAT", "ATATA"],
                            ["TTTTT", "ATATA"],
                            ["AAAAA", "AAAAT"],
                            ["AAAAA", "AAAAT"],
                        ]
                    )
                }
            ),
        ),
    ],
)
def test_similar_words(df, vocab, k, metric, res):
    assert_frame_equal(df.select(pds.query_similar_words("a", vocab, k=k, metric=metric)), res)

    assert_frame_equal(
        df.select(pds.query_similar_words("a", vocab, k=k, parallel=True, metric=metric)),
        res,
    )

    assert_frame_equal(
        df.lazy().select(pds.query_similar_words("a", vocab, k=k, metric=metric)).collect(),
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
    assert_frame_equal(df.select(pds.str_osa("a", pl.col("b"))), res)

    assert_frame_equal(df.select(pds.str_osa("a", pl.col("b"), parallel=True)), res)
    assert_frame_equal(df.lazy().select(pds.str_osa("a", pl.col("b"))).collect(), res)


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
    assert_frame_equal(df.select(pds.str_sorensen_dice("a", pl.col("b"))), res)

    assert_frame_equal(df.select(pds.str_sorensen_dice("a", pl.col("b"), parallel=True)), res)

    assert_frame_equal(df.lazy().select(pds.str_sorensen_dice("a", pl.col("b"))).collect(), res)


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
    assert_frame_equal(df.select(pds.str_jaccard("a", pl.col("b"), substr_size=size)), res)
    assert_frame_equal(
        df.select(pds.str_jaccard("a", pl.col("b"), substr_size=size, parallel=True)),
        res,
    )
    assert_frame_equal(
        df.lazy()
        .select(pds.str_jaccard("a", pl.col("b"), substr_size=size, parallel=True))
        .collect(),
        res,
    )


@pytest.mark.parametrize(
    "df, size, alpha, beta, res",
    [
        (
            pl.DataFrame({"a": ["apple", "test", "moon"], "b": ["let", "tests", "sun"]}),
            2,
            0.5,
            0.5,
            pl.DataFrame(
                {"a": pl.Series([0.3333333333333333, 0.8571428571428571, 0.0], dtype=pl.Float64)}
            ),
        ),
        (
            pl.DataFrame({"a": ["apple", "test", "moon"], "b": ["let", "tests", "sun"]}),
            3,
            0.1,
            0.9,
            pl.DataFrame({"a": pl.Series([0.0, 0.6896551724137931, 0.0], dtype=pl.Float64)}),
        ),
    ],
)
def test_tversky(df, size, alpha, beta, res):
    assert_frame_equal(
        df.select(pds.str_tversky_sim("a", pl.col("b"), alpha=alpha, beta=beta, substr_size=size)),
        res,
    )
    assert_frame_equal(
        df.select(
            pds.str_tversky_sim(
                "a", pl.col("b"), alpha=alpha, beta=beta, substr_size=size, parallel=True
            )
        ),
        res,
    )
    assert_frame_equal(
        df.lazy()
        .select(
            pds.str_tversky_sim(
                "a", pl.col("b"), alpha=alpha, beta=beta, substr_size=size, parallel=True
            )
        )
        .collect(),
        res,
    )


@pytest.mark.parametrize(
    "df, vocab, k, result",
    [
        (
            pl.DataFrame(
                {
                    "a": ["AAAA", "AAAB", "AABB", "AACC", "AADD"],
                }
            ),
            ["AAAD", "ACCC", "BBBB", "ABBB", "XXXX", "ADDA"],
            1,
            ["AAAD", "AAAD", "ABBB", "ACCC", "AAAD"],
        ),
    ],
)
def test_knn_str_lv(df, vocab, k, result):
    res = df.select(
        pds.query_similar_words(
            "a",
            vocab=vocab,
            k=k,
            metric="lv",
        ).alias("result")
    )
    res = res["result"].to_list()
    assert res == result
