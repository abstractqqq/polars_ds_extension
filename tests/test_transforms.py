import polars as pl
import polars_ds as pds
import polars_ds.transforms as t
import pytest
from polars.testing import assert_frame_equal


def test_target_encode():
    from category_encoders import TargetEncoder

    # random df to test target_encode
    df = pds.frame(size=2000).select(
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


def test_woe_encode():
    from category_encoders import WOEEncoder

    df = pds.frame(size=2000).select(
        pds.random_int(0, 3).cast(pl.String).alias("str_col"),
        pds.random_int(0, 2).alias("target"),
    )
    df1 = df.select(t.woe_encode(df, ["str_col"], target="target"))

    X = df["str_col"].to_numpy()
    y = df["target"].to_numpy()

    enc = WOEEncoder()
    df2 = pl.from_pandas(enc.fit_transform(X, y))
    df2.columns = ["str_col"]

    assert_frame_equal(df1, df2)


def test_scale():
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    df = pds.frame(size=100).select(
        pds.random().alias("a"),
        pds.random_normal(0.0, 1.0).alias("b"),
    )

    cols = df.columns
    df1 = df.with_columns(t.scale(df, cols=cols, method="standard"))

    scaler = StandardScaler()
    mat = scaler.fit_transform(df.to_pandas())
    df2 = pl.DataFrame(mat)
    df2.columns = cols

    assert_frame_equal(df1, df2)

    df1 = df.with_columns(t.scale(df, cols=cols, method="min_max"))

    scaler = MinMaxScaler()
    mat = scaler.fit_transform(df.to_pandas())
    df2 = pl.DataFrame(mat)
    df2.columns = cols

    assert_frame_equal(df1, df2)


@pytest.mark.parametrize(
    "df",
    [
        (pl.DataFrame({"a": [1, None, 2, 3, 3, 3, 3, 4], "b": [3, None, None, 3, 2, 2, 1, 4]})),
    ],
)
def test_impute(df):
    # all columns will be tested

    from sklearn.impute import SimpleImputer

    cols = df.columns
    df1 = df.with_columns(t.impute(df, cols=cols, method="mean"))

    imputer = SimpleImputer(strategy="mean")
    mat = imputer.fit_transform(df.to_pandas())
    df2 = pl.from_numpy(mat, schema=cols)

    assert_frame_equal(df1, df2)

    df1 = df.with_columns(t.impute(df, cols=cols, method="median"))

    imputer = SimpleImputer(strategy="median")
    mat = imputer.fit_transform(df.to_pandas())
    df2 = pl.from_numpy(mat, schema=cols)

    assert_frame_equal(df1, df2)


@pytest.mark.parametrize(
    "df, ranking",
    [
        (
            pl.DataFrame(
                {
                    "col": [
                        "bad",
                        "bad",
                        "good",
                        "neutral",
                        "neutral",
                        "neutral",
                        "bad",
                        "good",
                        None,
                        "unknown",
                    ]
                }
            ),
            ["bad", "neutral", "good"],
        ),
    ],
)
def test_rank_hot_encode(df, ranking):
    # all columns will be tested against the native Python lambda impl
    # Naive Python Lambda method
    mapping = dict(zip(ranking, range(len(ranking))))
    df_naive = df.select(
        (
            (
                pl.col("col").map_elements(
                    lambda x: mapping[x] if x in mapping else None,
                    return_dtype=pl.Int32,
                )
                >= i
            )
            .cast(pl.Int8)
            .fill_null(-1)
            .alias(f"col>={ranking[i]}")
            # fill_null with -1. If we get null, that means the value is not in ranking or is null.
        )
        for i in range(1, len(ranking))
    )

    df_to_test = df.select(t.rank_hot_encode(col="col", ranking=ranking))
    assert_frame_equal(df_to_test, df_naive)
