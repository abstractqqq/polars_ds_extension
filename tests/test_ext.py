import pytest
import polars as pl
from polars_ds.extensions import NumExt, StrExt  # noqa: F401
from polars.testing import assert_frame_equal

@pytest.mark.parametrize(
    "df, other, res",
    [
        (
            pl.DataFrame({
                "a": [1,2,3,4,5],
                "b": [1,2,2,2,10]
            }),
            3,
            pl.DataFrame({
                "a": [1,1,3,1,1]
            })
        ),
        (
            pl.DataFrame({
                "a": [1,2,3,4,5],
                "b": [1,2,2,2,10]
            }),
            pl.col("b"),
            pl.DataFrame({
                "a": [1,2,1,2,5]
            })
        ),
    ]

)
def test_gcd(df, other, res):
    
    assert_frame_equal(
        df.select(
            pl.col("a").num_ext.gcd(other)
        ),
        res
    )

    assert_frame_equal(
        df.lazy().select(
            pl.col("a").num_ext.gcd(other)
        ).collect(),
        res
    )

@pytest.mark.parametrize(
    "df, other, res",
    [
        (
            pl.DataFrame({
                "a": [1,2,3,4,5],
                "b": [1,2,2,2,10]
            }),
            3,
            pl.DataFrame({
                "a": [3,6,3,12,15]
            })
        ),
        (
            pl.DataFrame({
                "a": [1,2,3,4,5],
                "b": [1,2,2,2,10]
            }),
            pl.col("b"),
            pl.DataFrame({
                "a": [1,2,6,4,10]
            })
        ),
    ]

)
def test_lcm(df, other, res):
    
    assert_frame_equal(
        df.select(
            pl.col("a").num_ext.lcm(other)
        ),
        res
    )

    assert_frame_equal(
        df.lazy().select(
            pl.col("a").num_ext.lcm(other)
        ).collect(),
        res
    )

@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({
                "y":[1,0,1,1,1,0,0,1],
                "a": ["a", "b", "c", "a", "b", "c", "a", "a"]
            }),
            pl.DataFrame({
                "y": [0.6277411625893767]
            })
        ),
        (
            pl.DataFrame({
                "y":[1] * 8,
                "a": ["a", "b", "c", "a", "b", "c", "a", "a"]
            }),
            pl.DataFrame({
                "y": [-0.0]
            })
        ),
    ]
)
def test_cond_entropy(df, res):
    
    assert_frame_equal(
        df.select(
            pl.col("y").num_ext.cond_entropy(pl.col("a"))
        ),
        res
    )

    assert_frame_equal(
        df.lazy().select(
            pl.col("y").num_ext.cond_entropy(pl.col("a"))
        ).collect(),
        res
    )

# Hard to write generic tests because ncols can vary in X
def test_lstsq():

    df = pl.DataFrame({
        "y":[1,2,3,4,5],
        "a": [2,3,4,5,6],
        "b": [-1,-1,-1,-1,-1]
    })
    res = pl.DataFrame({
        "y": [[1.0, 1.0]]
    })
    assert_frame_equal(
        df.select(
            pl.col("y").num_ext.lstsq(pl.col("a"), pl.col("b"), add_bias = False)
        ),
        res
    )

    df = pl.DataFrame({
        "y":[1,2,3,4,5],
        "a": [2,3,4,5,6],
    })
    res = pl.DataFrame({
        "y": [[1.0, -1.0]]
    })
    assert_frame_equal(
        df.select(
            pl.col("y").num_ext.lstsq(pl.col("a"), add_bias = True)
        ),
        res
    )

@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({
                "a": ["thanks","thank","thankful"]
            }),
            pl.DataFrame({
                "a": ["thank","thank","thank"]
            })
        ),
        (
            pl.DataFrame({
                "a": ["playful","playing", "play", "played", "plays"]
            }),
            pl.DataFrame({
                "a": ["play","play", "play", "play", "play"]
            })
        ),
    ]
)
def test_snowball(df, res):
    
    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.snowball()
        ),
        res
    )

    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.snowball(parallel=True)
        ),
        res
    )

    assert_frame_equal(
        df.lazy().select(
            pl.col("a").str_ext.snowball()
        ).collect(),
        res
    )

@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({
                "a":["karolin", "karolin", "kathrin", "0000", "2173896"],
                "b":["kathrin", "kerstin", "kerstin", "1111", "2233796"]
            }),
            pl.DataFrame({
                "a": pl.Series([3,3,4,4,3], dtype=pl.UInt32)
            })
        ),
    ]
)
def test_hamming_dist(df, res):

    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.hamming_dist(pl.col("b"))
        )
        , res
    )
    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.hamming_dist(pl.col("b"), parallel=True)
        )
        , res
    )
    assert_frame_equal(
        df.lazy().select(
            pl.col("a").str_ext.hamming_dist(pl.col("b"))
        ).collect()
        , res
    )

@pytest.mark.parametrize(
    "df, res",
    [
        (
            pl.DataFrame({
                "a":["kitten", "mary", "may"],
                "b":["sitting", "merry", "mayer"]
            }),
            pl.DataFrame({
                "a": pl.Series([3,2,2], dtype=pl.UInt32)
            })
        ),
    ]
)
def test_levenshtein_dist(df, res):

    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.levenshtein_dist(pl.col("b"))
        )
        , res
    )

    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.levenshtein_dist(pl.col("b"), parallel=True)
        )
        , res
    )
    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.levenshtein_dist("may")
        )
        , pl.DataFrame({
            "a": pl.Series([6,1,0], dtype=pl.UInt32)
        })
    )
    assert_frame_equal(
        df.lazy().select(
            pl.col("a").str_ext.levenshtein_dist(pl.col("b"))
        ).collect()
        , res
    )

@pytest.mark.parametrize(
    "df, size, res",
    [
        (
            pl.DataFrame({
                "a":["apple", "test", "moon"],
                "b":["let", "tests", "sun"]
            })
            , 2
            , pl.DataFrame({
                "a": pl.Series([0.2,0.75,0.], dtype=pl.Float64)
            })
        ),
        (
            pl.DataFrame({
                "a":["apple", "test", "moon"],
                "b":["let", "tests", "sun"]
            })
            , 3
            , pl.DataFrame({
                "a": pl.Series([0.0, 2/3 , 0.0], dtype=pl.Float64)
            })
        ),
    ]
)
def test_str_jaccard(df, size, res):

    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.str_jaccard(pl.col("b"), substr_size=size)
        )
        , res
    )
    assert_frame_equal(
        df.select(
            pl.col("a").str_ext.str_jaccard(pl.col("b"), substr_size=size, parallel=True)
        )
        , res
    )
    assert_frame_equal(
        df.lazy().select(
            pl.col("a").str_ext.str_jaccard(pl.col("b"), substr_size=size, parallel=True)
        ).collect()
        , res
    )