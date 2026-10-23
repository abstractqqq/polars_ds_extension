import polars as pl
import polars_ds as pds
import polars_ds.pipeline.transforms as t
import pytest
from polars.testing import assert_frame_equal


def test_ordinal_encoding():
    from polars_ds.pipeline import Blueprint

    df_fit = pl.DataFrame({"gender": ["M", "F", "F", None], "data": [1, 2, 3, 4]})

    df_non_fit = pl.DataFrame({"gender": ["M", "X", "G", "F", None], "data": [1, 2, 3, 4, 5]})

    bp1 = Blueprint(df_fit, name="example", lowercase=True).ordinal_encode(
        cols=["gender"], null_value=-1, unknown_value=-2
    )

    pipe1 = bp1.materialize()
    result1_fit = pipe1.transform(df_fit)["gender"].to_list()
    # M = 1, F = 0, null = -1, unknown = -2
    assert result1_fit == [1.0, 0.0, 0.0, -1.0]
    result1_non_fit = pipe1.transform(df_non_fit)["gender"].to_list()
    assert result1_non_fit == [1.0, -2.0, -2.0, 0.0, -1.0]

    bp2 = Blueprint(df_fit, name="example", lowercase=True).ordinal_encode(cols=["gender"])
    # by default, nulls remain null
    pipe2 = bp2.materialize()
    result2 = pipe2.transform(df_fit)["gender"].to_list()
    assert result2 == [1.0, 0.0, 0.0, None]


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
            .alias(f"col>={ranking[i]}")
            # fill_null with -1. If we get null, that means the value is not in ranking or is null.
        )
        for i in range(1, len(ranking))
    )

    df_to_test = df.select(t.rank_hot_encode(col="col", ranking=ranking))
    assert_frame_equal(df_to_test, df_naive)


def test_impute_mode():
    df = pl.DataFrame({
        "a": [1, 2, 2, 3, None],
        "b": ["x", "y", "y", None, "z"],
    })
    res = df.with_columns(t.impute(df, cols=["a", "b"], method="mode"))
    assert res["a"].to_list() == [1, 2, 2, 3, 2]
    assert res["b"].to_list() == ["x", "y", "y", "y", "z"]


def test_conditional_impute_mode():
    df = pl.DataFrame({
        "a": [100, 200, 200, None, 999],
    })
    rule = {"a": pl.col("a").is_null() | (pl.col("a") > 500)}
    res = df.with_columns(t.conditional_impute(df, rules_dict=rule, method="mode"))
    assert res["a"].to_list() == [100, 200, 200, 200, 200]


def test_select_by_std_no_numeric():
    df = pl.DataFrame({"a": ["x", "y", "z"], "b": ["foo", "bar", "baz"]})
    res = t.select_by_std(df, ["a", "b"], min_=0.0, max_=10.0)
    # When no numeric columns exist, all columns should remain
    assert_frame_equal(df.select(res), df)


def test_enum_support():
    from polars_ds.pipeline import Blueprint

    df = pl.DataFrame({
        "color": pl.Series(["red", "blue", "red", None], dtype=pl.Enum(["red", "blue", "green"])),
        "target": [1.0, 0.0, 1.0, 0.0],
    })

    # One hot encode with Enum
    bp = Blueprint(df, name="enum_test").one_hot_encode(cols=["color"])
    pipe = bp.materialize()
    res = pipe.transform(df)
    assert "color_red" in res.columns
    assert "color_blue" in res.columns
    assert "color" not in res.columns

    # Ordinal encode with Enum (Enum sorts by enum definition order: red=0.0, blue=1.0)
    bp_ord = Blueprint(df, name="enum_ord").ordinal_encode(cols=["color"], null_value=-1)
    pipe_ord = bp_ord.materialize()
    res_ord = pipe_ord.transform(df)
    assert res_ord["color"].to_list() == [0.0, 1.0, 0.0, -1.0]


def test_explode_empty_as_null():
    from polars_ds.pipeline import Blueprint

    df = pl.DataFrame({
        "id": [1, 2, 3],
        "vals": [[10, 20], [], [30]],
    })

    # Default empty_as_null=False (drops empty list row)
    bp1 = Blueprint(df, name="explode_false").explode("vals", empty_as_null=False)
    pipe1 = bp1.materialize()
    res1 = pipe1.transform(df)
    assert len(res1) == 3
    assert res1["vals"].to_list() == [10, 20, 30]

    # empty_as_null=True (preserves row with null)
    bp2 = Blueprint(df, name="explode_true").explode("vals", empty_as_null=True)
    pipe2 = bp2.materialize()
    res2 = pipe2.transform(df)
    assert len(res2) == 4
    assert res2["vals"].to_list() == [10, 20, None, 30]

    # Serialization round-trip
    from polars_ds.pipeline import Pipeline
    pipe1_reloaded = Pipeline.from_json(pipe1.to_json())
    assert len(pipe1_reloaded.transform(df)) == 3

    pipe2_reloaded = Pipeline.from_json(pipe2.to_json())
    assert len(pipe2_reloaded.transform(df)) == 4

    # keep_nulls tests
    df_nulls = pl.DataFrame({
        "id": [1, 2],
        "vals": [[10, 20], None],
    })
    bp_keep = Blueprint(df_nulls, name="explode_keep").explode("vals", keep_nulls=True)
    assert len(bp_keep.materialize().transform(df_nulls)) == 3

    bp_drop = Blueprint(df_nulls, name="explode_drop").explode("vals", keep_nulls=False)
    assert len(bp_drop.materialize().transform(df_nulls)) == 2


def test_streaming_kwarg_normalization():
    from polars_ds.pipeline import Blueprint

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    bp = Blueprint(df, name="streaming_test").scale(["a"], method="standard")
    pipe = bp.materialize(streaming=True)
    res = pipe.transform(df, streaming=True)
    assert len(res) == 3

    pipe2 = bp.materialize(streaming=False)
    res2 = pipe2.transform(df, streaming=False)
    assert len(res2) == 3


def test_streaming_kwarg_normalization_v2(monkeypatch):
    import polars_ds.pipeline.pipeline as pipe_module
    from polars_ds.pipeline import Blueprint

    monkeypatch.setattr(pipe_module, "IS_POLARS_V2", True)
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    bp = Blueprint(df, name="streaming_test_v2").scale(["a"], method="standard")
    # In v2, streaming=True normalizes to engine='streaming'
    pipe = bp.materialize(streaming=True)
    res = pipe.transform(df, streaming=True)
    assert len(res) == 3

    pipe2 = bp.materialize(streaming=False)
    res2 = pipe2.transform(df, streaming=False)
    assert len(res2) == 3


def test_explode_explicit_empty_as_null():
    from polars_ds.pipeline import Blueprint

    df = pl.DataFrame({
        "id": [1, 2, 3],
        "vals": [[10, 20], [], [30]],
    })

    # Default preserves row count (empty_as_null=True)
    bp_default = Blueprint(df, name="exp_default").explode("vals")
    pipe_default = bp_default.materialize()
    res_default = pipe_default.transform(df)
    assert len(res_default) == 4
    assert res_default["vals"].to_list() == [10, 20, None, 30]

    # Explicit empty_as_null=False drops empty list row
    bp_drop = Blueprint(df, name="exp_drop").explode("vals", empty_as_null=False)
    pipe_drop = bp_drop.materialize()
    res_drop = pipe_drop.transform(df)
    assert len(res_drop) == 3
    assert res_drop["vals"].to_list() == [10, 20, 30]



def test_mode_impute_gated_by_is_polars_v2(monkeypatch):
    import polars_ds.pipeline.transforms as t_module

    df = pl.DataFrame({
        "a": [1, 2, 2, 3, None],
        "b": ["x", "y", "y", None, "z"],
    })

    # With IS_POLARS_V2 = True
    monkeypatch.setattr(t_module, "IS_POLARS_V2", True)
    res_v2 = df.with_columns(t_module.impute(df, cols=["a", "b"], method="mode"))
    assert res_v2["a"].to_list() == [1, 2, 2, 3, 2]
    assert res_v2["b"].to_list() == ["x", "y", "y", "y", "z"]

    # With IS_POLARS_V2 = False
    monkeypatch.setattr(t_module, "IS_POLARS_V2", False)
    res_v1 = df.with_columns(t_module.impute(df, cols=["a", "b"], method="mode"))
    assert res_v1["a"].to_list() == [1, 2, 2, 3, 2]
    assert res_v1["b"].to_list() == ["x", "y", "y", "y", "z"]



