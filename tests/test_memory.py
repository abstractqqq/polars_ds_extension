import polars as pl
import polars_ds  # noqa: F401
import numpy as np
import memray
# from scipy.stats import ks_2samp, ttest_ind

a = np.random.random(size=100_000)
b = np.random.random(size=100_000)
df = pl.DataFrame({"a": a, "b": b})


def lempel_ziv():
    _ = df.select((pl.col("a") > 0.1).num.lempel_ziv_complexity())


if __name__ == "__main__":
    with memray.Tracker("test.bin"):
        # lempel_ziv()
        df.select(pl.col("a") + pl.col("b"))
