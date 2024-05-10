import polars as pl
import polars_ds as pds
import polars_ds.transforms as t
from polars.testing import assert_frame_equal


def test_target_encode():
    from category_encoders import TargetEncoder

    # random df to test target_encode
    df = pds.random_data(size=2000, n_cols=0).select(
        pds.random_int(0, 3).cast(pl.String).alias("cat"), pds.random_int(0, 2).alias("target")
    )

    # polars_ds target encode
    df1 = df.select(
        t.target_encode(df, ["cat"], target="target", min_samples_leaf=20, smoothing=10.0)
    )

    # category_encoder target encode
    df_pd = df.to_pandas()
    y = df_pd["target"]
    cols = ["cat"]
    enc = TargetEncoder(min_samples_leaf=20, smoothing=10.0, cols=cols)
    enc.fit(df_pd[cols], y)
    df_transformed = enc.transform(df_pd[cols])
    df2 = pl.from_pandas(df_transformed[cols])  # output from category_encoders

    assert_frame_equal(df1, df2)
