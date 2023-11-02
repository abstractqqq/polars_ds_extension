import polars as pl
import timeit
from polars_ds.extensions import StrExt  # noqa: F401

def least_square1(df:pl.dataframe) -> pl.DataFrame:
    
    return df.select(
        pl.col("y").num_ext.lstsq(pl.col("a"), pl.col("b"))
    )

def least_square2(df:pl.dataframe) -> pl.DataFrame:
    
    return df.select(
        pl.col("y").num_ext.lstsq2(pl.col("a"), pl.col("b"))
    )


if __name__ == "__main__":
    df = pl.DataFrame({
        "a":pl.Series(range(500_000), dtype=pl.Float64),
        "b":pl.Series([1.0] * 500_000, dtype=pl.Float64),
        "y":pl.Series(range(500_000), dtype=pl.Float64) + 0.5,
    })

    res1 = least_square1(df)
    res2 = least_square2(df)

    from polars.testing import assert_frame_equal

    assert_frame_equal(
        res1, res2
    )

    time1 = timeit.timeit(lambda: least_square1(df), number = 10)
    time2 = timeit.timeit(lambda: least_square2(df), number = 10)
    print(f"Time for Implementation 1: {time1:.4f}s.")
    print(f"Time for Implementation 2: {time2:.4f}s.")