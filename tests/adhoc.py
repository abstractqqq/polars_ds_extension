import polars as pl
from polars_ds.extensions import StrExt  # noqa: F401

if __name__ == "__main__":
    df = pl.DataFrame({
        "a":["karolin", "karolin", "kathrin", "0000", "2173896"],
        "b":["kathrin", "kerstin", "kerstin", "1111", "2233796"]
    })


    res = df.select(
        pl.col("a").str_ext.hamming_dist(pl.col("b"))
    )

    print(res)