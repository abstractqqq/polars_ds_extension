"""Dtype, schema and row-preservation contracts across Polars versions."""

from itertools import product

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_ds as pds
from polars_ds.pipeline import Blueprint, Pipeline


@pytest.fixture(params=["auto", "streaming", "in-memory"])
def engine(request):
    return request.param


FLOAT_FUNCTIONS = [
    pds.gamma,
    pds.expit,
    pds.exp2,
    pds.logit,
    pds.fract,
    pds.trunc,
    pds.next_up,
    pds.next_down,
    pds.digamma,
]
DTYPES = [pl.Int32, pl.Int64, pl.UInt64, pl.Float32, pl.Float64]
if hasattr(pl, "Int128"):
    DTYPES.append(pl.Int128)


@pytest.mark.parametrize("fn", FLOAT_FUNCTIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_numeric_output_schema_and_nulls(fn, dtype, engine):
    df = pl.DataFrame({"x": pl.Series([1, 2, None], dtype=dtype)})
    query = df.lazy().select(fn("x"))
    result = query.collect(engine=engine)
    assert query.collect_schema() == result.schema
    assert result.to_series()[-1] is None


@pytest.mark.parametrize("fn", FLOAT_FUNCTIONS + [pds.query_first_digit_cnt])
def test_mixed_signed_unsigned_promotion(fn, engine):
    df = pl.DataFrame({"i": [1, 2, 3], "u": [1, 2, 3]}).with_columns(pl.col("u").cast(pl.UInt64))
    value = pl.col("i") + pl.col("u")
    actual = df.lazy().select(fn(value).alias("value")).collect(engine=engine)
    expected = df.lazy().select(fn(value.cast(pl.Float64)).alias("value")).collect(engine=engine)
    assert_frame_equal(actual, expected, check_dtypes=False)


@pytest.mark.skipif(not hasattr(pl, "Int128"), reason="Int128 added after minimum Polars")
def test_benford_int128_extremes(engine):
    values = [-(1 << 127), 10**30, -999, 0, None]
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int128)})
    result = df.lazy().select(pds.query_first_digit_cnt("x")).collect(engine=engine).item()
    counts = [0] * 9
    for value in values:
        if value:
            counts[int(str(abs(value))[0]) - 1] += 1
    assert result.to_list() == counts


@pytest.mark.parametrize("rows,cols", [(0, 2), (1, 2), (2, 3), (4, 2)])
@pytest.mark.parametrize("k", [1, 2])
def test_principal_component_schema_and_shape(rows, cols, k, engine):
    data = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    columns = [f"x{i}" for i in range(cols)]
    df = pl.DataFrame(data, schema=columns)
    query = df.lazy().select(pds.principal_components(*columns, k=k).alias("out"))
    result = query.collect(engine=engine)
    assert query.collect_schema() == result.schema
    assert result.height == rows
    assert result.unnest("out").columns == [f"pc{i + 1}" for i in range(k)]
    if rows >= k:
        expected = np.linalg.svd(data - data.mean(axis=0), compute_uv=False)[:k] ** 2
        actual = result.unnest("out").to_numpy()
        np.testing.assert_allclose((actual**2).sum(axis=0), expected, atol=1e-10)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("wide", [True, False])
def test_pca_complete_rows_and_wide_matrices(center, wide, engine):
    matrix = (
        np.array([[1.0, 3.0, 2.0], [2.0, 4.0, 7.0]]) if wide else np.array([[3.0, 7.0], [4.0, 8.0]])
    )
    cols = [f"x{i}" for i in range(matrix.shape[1])]
    clean = pl.DataFrame(matrix, schema=cols)
    invalid = pl.DataFrame(
        {c: [None, 900.0] if i == 0 else [700.0, None] for i, c in enumerate(cols)}
    )
    df = pl.concat([clean, invalid])
    result = (
        df.lazy()
        .select(pds.pca(*cols, center=center).alias("out"))
        .collect(engine=engine)
        .unnest("out")
    )
    if center:
        matrix = matrix - matrix.mean(axis=0)
    values = result["singular_value"].to_numpy()
    vectors = np.array(result["weight_vector"].to_list())
    assert vectors.shape == (min(matrix.shape), matrix.shape[1])
    np.testing.assert_allclose(values, np.linalg.svd(matrix, compute_uv=False), atol=1e-10)
    np.testing.assert_allclose(
        vectors.T @ np.diag(values**2) @ vectors, matrix.T @ matrix, atol=1e-9
    )


@pytest.mark.parametrize(
    "function", [pds.query_binary_metrics, pds.query_tpr_fpr, pds.query_confusion_matrix]
)
@pytest.mark.parametrize("has_positive", [True, False])
def test_classification_schemas(function, has_positive, engine):
    df = pl.DataFrame({"y": [0, int(has_positive), 0], "p": [0.1, 0.8, 0.3]})
    expr = (
        function("y", "p", all_metrics=True)
        if function is pds.query_confusion_matrix
        else function("y", "p")
    )
    query = df.lazy().select(expr.alias("out"))
    assert query.collect_schema() == query.collect(engine=engine).schema


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Float32, pl.Float64])
def test_survival_time_schema(dtype, engine):
    df = pl.DataFrame({"s": [1, 0, 1], "t": [1, 1, 2]}).with_columns(pl.col("t").cast(dtype))
    query = df.lazy().select(pds.query_kaplan_meier_prob("s", "t").alias("out"))
    result = query.collect(engine=engine)
    assert query.collect_schema() == result.schema
    np.testing.assert_allclose(result.unnest("out")["prob"].to_numpy(), [2 / 3, 0])


@pytest.mark.parametrize("two_columns", [True, False])
def test_pipeline_explode_preserves_rows_and_serialization(two_columns, engine):
    df = pl.DataFrame({"id": range(5), "x": [[], [1], None, [None], [2, 3]]})
    cols = ["x"]
    if two_columns:
        df = df.with_columns(pl.col("x").alias("y"))
        cols.append("y")
    pipe = Blueprint(df).explode(cols).materialize()
    restored = Pipeline.from_json(pipe.to_json())
    for candidate in [pipe, restored]:
        result = candidate.transform(df, return_lazy=True).collect(engine=engine)
        assert result["id"].to_list() == [0, 1, 2, 3, 4, 4]


@pytest.mark.parametrize("x,y", [([], [1]), (None, [1]), ([1, 2], [3])])
def test_pipeline_explode_rejects_mismatched_lengths(x, y, engine):
    df = pl.DataFrame(
        {
            "x": pl.Series([x], dtype=pl.List(pl.Int64)),
            "y": pl.Series([y], dtype=pl.List(pl.Int64)),
        }
    )
    pipe = Blueprint(df).explode(["x", "y"]).materialize()
    with pytest.raises(pl.exceptions.ShapeError):
        pipe.transform(df, return_lazy=True).collect(engine=engine)


# Polars 2 promotes mixed signed integers/UInt64 to Int128 instead of Float64.
SIGNED_UNSIGNED_COORDS = [
    types
    for types in product([pl.Int32, pl.Int64, pl.UInt64], repeat=4)
    if pl.UInt64 in types and any(t in (pl.Int32, pl.Int64) for t in types)
]


@pytest.mark.parametrize("dtypes", SIGNED_UNSIGNED_COORDS)
def test_haversine_mixed_integer_coordinates(dtypes, engine):
    columns = ["lat", "lon", "end_lat", "end_lon"]
    values = [[0, 30, None], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    df = pl.DataFrame({c: pl.Series(v, dtype=t) for c, v, t in zip(columns, values, dtypes)})
    query = df.lazy().select(pds.haversine(*columns).alias("km"))
    result = query.collect(engine=engine)
    assert query.collect_schema() == result.schema == {"km": pl.Float64}
    assert result["km"][2] is None
    np.testing.assert_allclose(result["km"][:2].to_numpy(), [0, 6371 * np.pi / 6])


@pytest.mark.parametrize("context", ["group", "over"])
@pytest.mark.parametrize("broadcast_end", [False, True])
def test_haversine_promoted_coordinates_in_groups(context, broadcast_end, engine):
    df = pl.DataFrame(
        {
            "g": [0, 1, 0],
            "lat": pl.Series([0, 30, None], dtype=pl.Int64),
            "lon": pl.Series([0, 0, 0], dtype=pl.UInt64),
        }
    )
    end_lat = pl.lit(0, dtype=pl.Int64) if broadcast_end else pl.col("lon")
    end_lon = pl.lit(0, dtype=pl.UInt64) if broadcast_end else pl.col("lon")
    expr = pds.haversine("lat", "lon", end_lat, end_lon).alias("km")
    query = (
        df.lazy().group_by("g").agg(expr).sort("g")
        if context == "group"
        else df.lazy().select(expr.over("g"))
    )
    result = query.collect(engine=engine)
    if context == "group" and broadcast_end and pl.__version__.startswith("1.4."):
        # Polars 1.4 infers a scalar for grouped elementwise plugins with
        # scalar arguments, even though their runtime result is a list.
        assert query.collect_schema()["km"] == pl.Float64
        assert result.schema["km"] == pl.List(pl.Float64)
    else:
        assert query.collect_schema() == result.schema
    if context == "group":
        assert result["km"][0].to_list() == [0.0, None]
        assert result["km"][1][0] == pytest.approx(6371 * np.pi / 6)
    else:
        assert result["km"][2] is None
        np.testing.assert_allclose(result["km"][:2].to_numpy(), [0, 6371 * np.pi / 6])


CATEGORY_DTYPES = [pl.String, pl.Int64, pl.Boolean, pl.Categorical, pl.Enum(["a", "b"])]


@pytest.mark.parametrize("dtype", CATEGORY_DTYPES)
@pytest.mark.parametrize("grouped", [False, True])
def test_discrete_psi_report_schema_and_values(dtype, grouped, engine):
    actual = [0, 0, 1, 1] * 2
    baseline = [0, 1, 1, 1] * 2
    if dtype not in (pl.Int64, pl.Boolean):
        actual = ["ab"[i] for i in actual]
        baseline = ["ab"[i] for i in baseline]
    elif dtype == pl.Boolean:
        actual = list(map(bool, actual))
        baseline = list(map(bool, baseline))
    df = pl.DataFrame(
        {
            "actual": pl.Series(actual, dtype=dtype),
            "baseline": pl.Series(baseline, dtype=dtype),
            "g": [0] * 4 + [1] * 4,
        }
    )
    report = pds.psi_discrete("actual", "baseline", return_report=True).alias("report")
    scalar = pds.psi_discrete("actual", "baseline").alias("psi")
    queries = [
        df.lazy().group_by("g").agg(expr).sort("g") if grouped else df.lazy().select(expr)
        for expr in [report, scalar]
    ]
    result, summary = [query.collect(engine=engine) for query in queries]
    assert queries[0].collect_schema() == result.schema
    assert queries[1].collect_schema() == summary.schema
    expected = (0.5 - 0.25) * np.log(0.5 / 0.25) + (0.5 - 0.75) * np.log(0.5 / 0.75)
    np.testing.assert_allclose(summary["psi"].to_numpy(), expected)
    flat = result.explode("report").unnest("report") if grouped else result.unnest("report")
    # Polars 1.4's FFI exports Enum as Categorical; preserve that legacy output.
    category_dtype = (
        pl.Categorical
        if isinstance(dtype, pl.Enum) and pl.__version__.startswith("1.4.")
        else dtype
    )
    assert flat.schema["baseline_category"] == category_dtype
    assert flat.schema["actual_category"] == category_dtype
    assert flat.height == (4 if grouped else 2)
    if not grouped:
        # Field projection must also resolve using the plugin's planned schema.
        categories = queries[0].select(pl.col("report").struct.field("baseline_category"))
        assert categories.collect(engine=engine).height == 2


@pytest.mark.parametrize("use_literals", [False, True])
def test_discrete_psi_empty_report_schema(use_literals, engine):
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.String)})
    operand = pl.lit(pl.Series([], dtype=pl.String)) if use_literals else pl.col("a")
    query = df.lazy().select(pds.psi_discrete(operand, operand, return_report=True).alias("report"))
    result = query.collect(engine=engine)
    assert query.collect_schema() == result.schema
    assert result.height == 0
    assert result.unnest("report").columns == [
        "baseline_category",
        "actual_category",
        "baseline_pct",
        "actual_pct",
        "psi_bin",
    ]


@pytest.mark.parametrize("context", ["select", "group", "over"])
@pytest.mark.parametrize("use_aliases", [False, True])
def test_chi2_full_report_preserves_names_in_groups(context, use_aliases, engine):
    from scipy.stats import chi2_contingency

    df = pl.DataFrame(
        {
            "a": ["a", "a", "a", "b", "b", "b"] * 2,
            "b": ["x", "x", "y", "x", "y", "y"] * 2,
            "g": [0] * 6 + [1] * 6,
        }
    )
    names = ["left", "right"] if use_aliases else ["a", "b"]
    args = [pl.col(c).alias(n) for c, n in zip(["a", "b"], names)]
    expr = pds.chi2(*args, return_full=True).alias("report")
    if context == "group":
        query = df.lazy().group_by("g").agg(expr).sort("g")
    elif context == "over":
        query = df.lazy().select(expr.over("g", mapping_strategy="join"))
    else:
        query = df.lazy().select(expr)
    result = query.collect(engine=engine)
    if context == "over" and pl.__version__.startswith("1.4."):
        # Polars 1.4's planner reports the inner dtype for window joins,
        # including pure Polars expressions. Its runtime correctly returns List.
        assert query.collect_schema()["report"] == result.schema["report"].inner
    else:
        assert query.collect_schema() == result.schema
    flat = result if context == "select" else result.explode("report")
    flat = flat.unnest("report")
    table = np.array([[2, 1], [1, 2]]) * (2 if context == "select" else 1)
    expected = chi2_contingency(table, correction=False)
    np.testing.assert_allclose(flat["statistic"].to_numpy(), expected.statistic)
    np.testing.assert_allclose(flat["pvalue"].to_numpy(), expected.pvalue)
    np.testing.assert_allclose(flat["E[freq]"].to_numpy(), 3 if context == "select" else 1.5)
    assert flat["dof"].to_list() == [1] * flat.height
    assert flat.select(names).unique().height == 4
