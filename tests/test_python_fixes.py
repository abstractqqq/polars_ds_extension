"""
Regression tests for bugs in the Python layer of polars_ds.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polars_ds as pds
from polars_ds.linear_models import LR
from polars_ds.pipeline import Blueprint
from polars_ds.sample_and_split import downsample


# --- gmean / weighted_gmean: previously called the non-existent Expr.ln() ---


def test_gmean():
    values = [1.0, 4.0, 2.0, 3.0, 5.0]
    df = pl.DataFrame({"a": values})
    result = df.select(pds.gmean("a")).item()
    expected = math.prod(values) ** (1 / len(values))
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("is_normalized", [True, False])
def test_weighted_gmean(is_normalized: bool):
    values = np.array([1.0, 4.0, 2.0, 3.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    if not is_normalized:
        weights = weights * 10.0  # same relative weights, sum != 1

    df = pl.DataFrame({"a": values, "w": weights})
    result = df.select(pds.weighted_gmean("a", "w", is_normalized=is_normalized)).item()
    expected = np.exp(np.dot(np.log(values), weights / weights.sum()))
    assert result == pytest.approx(expected)


# --- random_int / random bound handling ---


def test_random_int_accepts_expr_bounds():
    # `lower == upper` on two expressions used to raise "ambiguous truth value"
    df = pds.frame(size=1000).select(pds.random_int(pl.lit(3), pl.lit(10), seed=42).alias("r"))
    r = df["r"]
    assert r.min() >= 3
    assert r.max() < 10


@pytest.mark.parametrize("lower, upper", [(5, 5), (6, 5)])
def test_random_int_invalid_int_bounds(lower: int, upper: int):
    with pytest.raises(ValueError):
        pds.random_int(lower, upper)


def test_random_accepts_int_bounds():
    df = pds.frame(size=1000).select(pds.random(2, 3, seed=42).alias("r"))
    r = df["r"]
    assert r.dtype == pl.Float64
    assert r.min() >= 2.0
    assert r.max() < 3.0


# --- similar_to_vocab(metric="jw"): previously passed an unsupported kwarg to str_jw ---


def test_similar_to_vocab_jw():
    df = pl.DataFrame({"s": ["hello", "hallo", "xyz"]})
    result = df.select(pds.similar_to_vocab("s", ["hello"], threshold=0.8, metric="jw"))
    assert result["s"].to_list() == [True, True, False]


# --- LR.fit_df(show_report=True): previously imported a non-existent function ---


@pytest.mark.parametrize("has_bias", [True, False])
def test_lr_fit_df_show_report(has_bias: bool, capsys):
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.1, 3.9, 6.2, 7.8, 10.1]})
    model = LR(has_bias=has_bias).fit_df(df, ["x"], "y", show_report=True)
    assert model.is_fit()

    out = capsys.readouterr().out
    assert "beta" in out
    # The report must agree with the model about the bias term
    assert ("__bias__" in out) == has_bias


# --- Pipeline.with_features_out: previously returned None ---


def test_pipeline_with_features_out_is_chainable():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    pipe = Blueprint(df, name="test").scale(["a"]).materialize()
    returned = pipe.with_features_out(["a"])
    assert returned is pipe
    assert pipe.feature_names_out_ == ["a"]


# --- downsample: a single (expr, value) tuple used to be iterated element by element ---


@pytest.mark.parametrize("lazy", [False, True])
def test_downsample_single_condition_tuple(lazy: bool):
    df = pl.DataFrame({"g": ["A"] * 10 + ["B"] * 10, "v": range(20)})
    frame = df.lazy() if lazy else df

    out = downsample(frame, (pl.col("g") == "A", 3), seed=42, return_df=True)
    assert isinstance(out, pl.DataFrame)
    counts = dict(out.group_by("g").len().iter_rows())
    assert counts == {"A": 3, "B": 10}


def test_downsample_single_tuple_matches_list():
    df = pl.DataFrame({"g": ["A"] * 10 + ["B"] * 10, "v": range(20)})
    cond = (pl.col("g") == "A", 0.5)
    single = downsample(df, cond, seed=7).sort("v")
    listed = downsample(df, [cond], seed=7).sort("v")
    assert single.equals(listed)
