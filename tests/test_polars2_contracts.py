"""Dtype, schema and row-preservation contracts across Polars versions."""

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
