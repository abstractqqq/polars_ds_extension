import pytest
import polars as pl


@pytest.mark.parametrize(
    "df, res",
    (
        pl.DataFrame({
            "a": [1,2,3,4,5]
        }),
        pl.DataFrame({
            "a": [1,1,3,1,1]
        })
    )

)
def test_gcd(df:pl.DataFrame, res:pl.DataFrame):
    pass