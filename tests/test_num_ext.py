import pytest
import polars as pl
from polars_ds.extensions import NumExt  # noqa: F401
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