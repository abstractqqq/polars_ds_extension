"""Regression cases for Polars 2 streaming and statistical compatibility.

Run with POLARS_MAX_THREADS=144 to exercise single-row streaming morsels.
The cases also run on Polars 1.44 to protect the preceding version's behavior.
"""

from functools import partial

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.stats import ks_2samp

import polars_ds as pds
from polars_ds.exprs import string


@pytest.fixture(params=["auto", "streaming", "in-memory"])
def engine(request):
    return request.param


STRING_FUNCTIONS = [
    (string.str_hamming, pl.UInt32),
    (partial(string.str_hamming, pad=True), pl.UInt32),
    (string.str_jaro, pl.Float64),
    (string.str_jw, pl.Float64),
    (string.str_leven, pl.UInt32),
    (partial(string.str_leven, return_sim=True), pl.Float64),
    (string.str_d_leven, pl.UInt32),
    (partial(string.str_d_leven, return_sim=True), pl.Float64),
    (string.str_lcs_substr, pl.String),
    (string.str_lcs_subseq, pl.String),
    (string.str_lcs_subseq_dist, pl.Float64),
    (partial(string.str_lcs_subseq_dist, return_sim=False), pl.UInt32),
    (string.str_osa, pl.UInt32),
    (partial(string.str_osa, return_sim=True), pl.Float64),
    (string.str_fuzz, pl.Float64),
    (string.str_jaccard, pl.Float64),
    (string.str_overlap_coeff, pl.Float64),
    (string.str_sorensen_dice, pl.Float64),
    (partial(string.str_tversky_sim, alpha=0.5, beta=0.5), pl.Float64),
    (partial(string.filter_by_hamming, bound=2), pl.Boolean),
    (partial(string.filter_by_levenshtein, bound=2), pl.Boolean),
]


@pytest.mark.parametrize("fn,dtype", STRING_FUNCTIONS)
@pytest.mark.parametrize("length", [1, 50])
def test_null_string_scalar_propagates(fn, dtype, length, engine):
    df = pl.DataFrame({"s": ["abc"] * length})
    query = df.lazy().select(fn("s", pl.lit(None, dtype=pl.String)).alias("result"))
    assert query.collect_schema()["result"] == dtype
    out = query.collect(engine=engine)["result"]
    assert out.dtype == dtype
    assert out.to_list() == [None] * length


@pytest.mark.parametrize("fn,dtype", STRING_FUNCTIONS)
def test_null_string_column_in_single_row_batches(fn, dtype, engine):
    df = pl.DataFrame({"s": ["abc"] * 49 + [None], "other": ["abc"] * 49 + [None]})
    result = df.lazy().select(fn("s", "other").alias("result")).collect(engine=engine)
    assert result["result"].dtype == dtype
    assert result["result"].null_count() == 1
    assert result["result"][-1] is None


@pytest.mark.parametrize("fn", [pds.gcd, pds.lcm])
def test_null_integer_scalar_propagates(fn, engine):
    df = pl.DataFrame({"x": range(50)})
    out = df.lazy().select(fn("x", pl.lit(None, dtype=pl.Int32))).collect(engine=engine)
    assert out.to_series().dtype == pl.Int32
    assert out.to_series().to_list() == [None] * 50


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize("null_latitude", [False, True])
def test_null_haversine_scalar_propagates(dtype, null_latitude, engine):
    df = pl.DataFrame(
        {"lat": [0.0] * 50, "lon": [0.0] * 50},
        schema_overrides={
            "lat": dtype,
            "lon": dtype,
        },
    )
    ylat = pl.lit(None if null_latitude else 1.0, dtype=dtype)
    ylon = pl.lit(1.0 if null_latitude else None, dtype=dtype)
    out = df.lazy().select(pds.haversine("lat", "lon", ylat, ylon)).collect(engine=engine)
    assert out.to_series().dtype == dtype
    assert out.to_series().to_list() == [None] * 50


def test_ks_multichunk_matches_scipy(engine):
    a = np.arange(1000, dtype=float)
    b = a + 17.5
    df = pl.concat(
        [pl.DataFrame({"a": a[:500], "b": b[:500]}), pl.DataFrame({"a": a[500:], "b": b[500:]})],
        rechunk=False,
    )
    assert df["a"].n_chunks() == 2
    out = df.lazy().select(pds.ks_2samp("a", "b").struct.field("statistic")).collect(engine=engine)
    assert out.item() == pytest.approx(ks_2samp(a, b).statistic)


def test_streaks_across_chunks(engine):
    df = pl.concat(
        [pl.DataFrame({"x": [False, True, True]}), pl.DataFrame({"x": [True, False, True]})],
        rechunk=False,
    )
    out = (
        df.lazy()
        .select(pds.query_longest_streak("x"), pds.query_avg_streak("x"))
        .collect(engine=engine)
    )
    assert out.row(0) == (3, 2.0)
    streaks = df.lazy().select(pds.query_streak("x")).collect(engine=engine)
    assert streaks.to_series().to_list() == [1, 3, 1, 1]


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("with_nulls", [False, True])
@pytest.mark.parametrize("as_ratio", [False, True])
def test_explained_variance_uses_complete_observations(center, with_nulls, as_ratio, engine):
    rows = [[1.0, 2.0], [2.0, 0.0], [4.0, 1.0], [5.0, 4.0]]
    df = pl.DataFrame(rows, schema=["x", "y"], orient="row")
    if with_nulls:
        df = pl.concat([df, pl.DataFrame({"x": [None, 900.0], "y": [800.0, None]})])
    matrix = np.array(rows)
    if center:
        matrix = matrix - matrix.mean(axis=0)
    expected = np.linalg.svd(matrix, compute_uv=False) ** 2 / (len(rows) - 1)
    if as_ratio:
        expected /= expected.sum()
    out = (
        df.lazy()
        .select(
            pds.singular_values("x", "y", center=center, as_explained_var=True, as_ratio=as_ratio)
        )
        .collect(engine=engine)
    )
    np.testing.assert_allclose(out.item().to_numpy(), expected)


@pytest.mark.parametrize(
    "matrix", [[[1.0], [2.0], [3.0]], [[1.0, 0.0, 2.0, -1.0, 3.0], [2.0, 1.0, 0.0, 3.0, 1.0]]]
)
def test_explained_variance_matrix_shapes(matrix, engine):
    columns = [f"x{i}" for i in range(len(matrix[0]))]
    df = pl.DataFrame(matrix, schema=columns, orient="row")
    values = np.array(matrix)
    expected = np.linalg.svd(values - values.mean(axis=0), compute_uv=False) ** 2
    expected /= len(values) - 1
    out = (
        df.lazy()
        .select(pds.singular_values(*columns, as_explained_var=True))
        .collect(engine=engine)
        .item()
        .to_numpy()
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize("as_explained_var", [False, True])
@pytest.mark.parametrize("as_ratio", [False, True])
def test_explained_variance_per_group(as_explained_var, as_ratio, engine):
    df = pl.DataFrame({"g": [0] * 4 + [1] * 6, "x": range(10), "y": [0, 1] * 5})
    result = (
        df.lazy()
        .group_by("g")
        .agg(
            pds.singular_values(
                "x", "y", as_explained_var=as_explained_var, as_ratio=as_ratio
            ).alias("variance")
        )
        .collect(engine=engine)
        .sort("g")
    )
    for g, values in result.iter_rows():
        matrix = df.filter(pl.col("g") == g).select("x", "y").to_numpy()
        expected = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
        if as_explained_var:
            expected = expected**2 / (len(matrix) - 1)
        if as_ratio:
            expected /= expected.sum()
        np.testing.assert_allclose(values, expected)


@pytest.mark.parametrize("length", [0, 1])
def test_svd_insufficient_observations(length, engine):
    df = pl.DataFrame({"x": pl.Series([2.0] * length, dtype=pl.Float64)})
    out = df.lazy().select(pds.singular_values("x", as_explained_var=True)).collect(engine=engine)
    assert out.item() is None
    values = df.lazy().select(pds.singular_values("x")).collect(engine=engine).item()
    assert values.to_list() == ([0.0] if length else [])


def test_svd_all_null_observations(engine):
    df = pl.DataFrame(
        {"x": [None, None], "y": [1.0, None]},
        schema_overrides={
            "x": pl.Float64,
            "y": pl.Float64,
        },
    )
    query = df.lazy().select(pds.singular_values("x", "y"))
    assert query.collect(engine=engine).item().to_list() == []


def test_svd_requires_features():
    with pytest.raises(ValueError, match="at least one feature"):
        pds.singular_values()


@pytest.mark.parametrize("explained", [False, True])
def test_svd_preserves_output_name(explained, engine):
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 0.0, 2.0]})
    for feature, expected_name in [("x", "x"), (pl.col("x").alias("feature"), "feature")]:
        query = df.lazy().select(pds.singular_values(feature, "y", as_explained_var=explained))
        assert query.collect(engine=engine).columns == [expected_name]


def test_linear_quantile_psi_parity(engine):
    df = pl.DataFrame({"ref": np.arange(80, dtype=float), "new": np.arange(80) + 0.25})
    report = (
        df.lazy()
        .select(pds.psi("new", "ref", n_bins=4, return_report=True))
        .collect(engine=engine)
        .to_series()
        .struct.unnest()
    )
    assert report["<="].to_list() == [19.75, 39.5, 59.25, float("inf")]
    np.testing.assert_allclose(report["baseline_pct"].to_numpy(), [0.25] * 4)
    np.testing.assert_allclose(report["actual_pct"].to_numpy(), [0.25] * 4)
    assert report["psi_bin"].sum() == pytest.approx(0.0)


def test_woe_iv_preserve_intervals_and_values(engine):
    df = pl.DataFrame({"x": np.arange(80, dtype=float), "target": [0] * 40 + [1] * 40})
    expected_labels = ["(-inf, 19.75]", "(19.75, 39.5]", "(39.5, 59.25]", "(59.25, inf]"]
    woe = df.lazy().select(pds.woe("x", "target", n_bins=4)).collect(engine=engine)
    iv = (
        df.lazy()
        .select(pds.info_value("x", "target", n_bins=4, return_sum=False))
        .collect(engine=engine)
    )
    woe = woe.to_series().struct.unnest().sort("value")
    iv = iv.to_series().struct.unnest().sort("value")
    assert woe["value"].to_list() == expected_labels
    assert iv["value"].to_list() == expected_labels
    expected_woe = np.log(np.array([1, 1, 21, 21]) / np.array([21, 21, 1, 1]))
    np.testing.assert_allclose(woe["woe"].to_numpy(), expected_woe)
    expected_iv = np.array([-20, -20, 20, 20]) / 42 * expected_woe
    np.testing.assert_allclose(iv["iv"].to_numpy(), expected_iv)
    # A non-finite feature must remove its corresponding target as well.
    invalid = pl.DataFrame({"x": [None, float("nan"), float("inf")], "target": [1, 0, 1]})
    dirty = pl.concat([df, invalid])
    out = dirty.lazy().select(pds.woe("x", "target", n_bins=4)).collect(engine=engine)
    assert_frame_equal(out.to_series().struct.unnest().sort("value"), woe)
