import polars as pl
import polars_ds as pds
import polars_ds.modeling.transforms as t
import pytest
from polars.testing import assert_frame_equal


def test_linear_impute():
    df = pl.DataFrame(
        {
            "a": [3, 2, 3, 4, 5, 6, 7, 8, 9, 11],
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "c": [4.0, 4.0, None, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0],
        }
    )

    imputed_c = df.with_columns(
        c=t.linear_impute(df, features=["a", "b"], target="c")[0].cast(pl.Float64)
    ).select("c")

    correct_c = df.select(c=(pl.col("a") + pl.col("b")).cast(pl.Float64))

    assert_frame_equal(imputed_c, correct_c)


def test_conditional_impute():
    df = pl.DataFrame(
        {
            "a": [float("nan"), None, float("inf"), 9999, 100, 100, 100, 800],
        }
    )

    res = df.with_columns(
        t.conditional_impute(
            df,
            {"a": ((pl.col("a").is_finite().not_()) | pl.col("a").is_null() | (pl.col("a") > 899))},
            method="mean",
        )[0].alias("result")
    )["result"]

    assert list(res)[:4] == [275.0, 275.0, 275.0, 275.0]

    res = df.with_columns(
        t.conditional_impute(
            df,
            {"a": ((pl.col("a").is_finite().not_()) | pl.col("a").is_null() | (pl.col("a") > 899))},
            method="median",
        )[0].alias("result")
    )["result"]

    assert list(res)[:4] == [100.0, 100.0, 100.0, 100.0]


def test_winsorize():
    df = pds.frame(size=1000).select(
        pds.random(0.0, 1.0).alias("x1"),
    )

    q_low = 0.05
    q_high = 0.95

    should_be_true = (
        df.with_columns(x2=t.winsorize(df, ["x1"], q_low=q_low, q_high=q_high)[0])
        .select(
            # the max and min of x2 should be 0.95, 0.05 - percentile of x1
            x1=pl.col("x2").max() == pl.col("x1").quantile(q_high),
            x2=pl.col("x2").min() == pl.col("x1").quantile(q_low),
        )
        .row(0)
    )

    assert all(should_be_true)


def test_polynomial_features_deg2():
    df = pds.frame(size=1000).select(
        pds.random(0.0, 1.0).alias("x1"),
        pds.random(0.0, 1.0).alias("x2"),
    )

    df_test = df.select(
        result=t.polynomial_features(["x1", "x2"], degree=2, interaction_only=True)[0]
    )

    df_correct = df.select(result=pl.col("x1") * pl.col("x2"))

    assert_frame_equal(df_test, df_correct)

    df_test = df.select(t.polynomial_features(["x1", "x2"], degree=2))

    df_correct = df.select(
        (pl.col("x1") * pl.col("x1")).alias("x1*x1"),
        (pl.col("x1") * pl.col("x2")).alias("x1*x2"),
        (pl.col("x2") * pl.col("x2")).alias("x2*x2"),
    )

    assert_frame_equal(df_test, df_correct)


def test_robust_scale():
    q_low = 0.25
    q_high = 0.75

    df = pds.frame(size=1000).select(
        pds.random(0.0, 1.0).alias("x1"),
    )

    method = "midpoint"

    robust_x1 = df.select(
        x2=t.robust_scale(df, ["x1"], q_low=q_low, q_high=q_high, method=method)[0]
    )

    robust_correct = df.select(
        x2=(pl.col("x1") - pl.col("x1").quantile(q_low, interpolation=method))
        / (
            pl.col("x1").quantile(q_high, interpolation=method)
            - pl.col("x1").quantile(q_low, interpolation=method)
        )
    )

    assert_frame_equal(robust_x1, robust_correct)


def test_one_hot_encode_options():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [None, "a", "a", "c", "d", "a", "a", None, "b", "c"],
        }
    )

    # polars_ds's one hot encode transform ignores nulls all the time
    # if you want nulls to be one-hot-encoded, use col('abc').is_null()

    df_t1 = df.with_columns(t.one_hot_encode(df, cols=["b"]))

    columns = df_t1.columns
    assert columns == ["a", "b", "b_a", "b_b", "b_c", "b_d"]  # the new columns are in lex order

    df_t2 = df.with_columns(t.one_hot_encode(df, cols=["b"], separator="|"))
    columns = df_t2.columns
    assert columns == ["a", "b", "b|a", "b|b", "b|c", "b|d"]  # the new columns are in lex order

    df_t3 = df.with_columns(t.one_hot_encode(df, cols=["b"], separator="|", drop_first=True))
    columns = df_t3.columns
    assert columns == ["a", "b", "b|b", "b|c", "b|d"]  # Drops the first


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

    # Test again with different min_sample_leaf and smoothing

    # random df to test target_encode
    df = pds.frame(size=2000).select(
        pds.random_int(0, 3).cast(pl.String).alias("cat"), pds.random_int(0, 2).alias("target")
    )

    # polars_ds target encode
    df1 = df.select(
        t.target_encode(df, ["cat"], target="target", min_samples_leaf=30, smoothing=5.0)
    )

    # category_encoder target encode
    df_pd = df.to_pandas()
    y = df_pd["target"]
    cols = ["cat"]
    enc = TargetEncoder(min_samples_leaf=30, smoothing=5.0, cols=cols)
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
