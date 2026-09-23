"""Finite-window EWLS against independent, per-window weighted least squares."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_ds as pds
import polars_ds.config as cfg


@pytest.fixture(params=[True, False], ids=["f64", "f32"])
def precision(request, monkeypatch):
    monkeypatch.setattr(cfg, "LIN_REG_EXPR_F64", request.param)
    return request.param


def data(n=180, p=3, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    y = x @ np.arange(1, p + 1) + 0.7 + rng.normal(scale=0.3, size=n)
    return pl.DataFrame({**{f"x{i}": x[:, i] for i in range(p)}, "y": y})


def reference(df, columns, window, half_life, bias, min_rows, indices=None):
    # Quantize inputs exactly as the selected plugin does, then use independent
    # Float64 SVD solves. Ages come from original positions, before masking.
    dtype = np.float64 if cfg.LIN_REG_EXPR_F64 else np.float32
    x = df.select(columns).to_numpy().astype(dtype).astype(np.float64)
    y = df["y"].to_numpy().astype(dtype).astype(np.float64)
    if bias:
        x = np.column_stack([x, np.ones(len(df))])
    out = {}
    for t in range(len(df)) if indices is None else indices:
        if t < window - 1:
            out[t] = None
            continue
        xx, yy = x[t - window + 1 : t + 1], y[t - window + 1 : t + 1]
        valid = np.isfinite(xx).all(axis=1) & np.isfinite(yy)
        if valid.sum() < max(min_rows, x.shape[1]):
            out[t] = None
            continue
        ages = np.arange(window - 1, -1, -1, dtype=float)[valid]
        # A common positive weight factor does not change least squares. Shift
        # ages before exponentiation so an entirely tiny window remains usable.
        w = np.exp2(-(ages - ages.min()) / half_life)
        beta, _, rank, _ = np.linalg.lstsq(
            xx[valid] * np.sqrt(w)[:, None], yy[valid] * np.sqrt(w), rcond=None
        )
        out[t] = beta if rank == x.shape[1] else None
    return out


def assert_reference(df, result, columns, window, half_life, bias, min_rows, indices=None):
    expected = reference(df, columns, window, half_life, bias, min_rows, indices)
    # Native f32 solves use the same relative tolerance as the existing LR tests;
    # f64 keeps the tighter EWLS accuracy requirement.
    rtol, atol = (1e-8, 1e-10) if cfg.LIN_REG_EXPR_F64 else (1e-4, 1e-5)
    for t, beta in expected.items():
        got = result["coeffs"][t]
        if beta is None:
            assert got is None, t
            assert result["pred"][t] is None, t
        else:
            assert got is not None, t
            np.testing.assert_allclose(got.to_numpy(), beta, rtol=rtol, atol=atol)
            x = np.asarray(df.select(columns).row(t), dtype=float)
            if not np.isfinite(x).all():
                assert result["pred"][t] is None, t
            else:
                if not cfg.LIN_REG_EXPR_F64:
                    x = x.astype(np.float32).astype(np.float64)
                if bias:
                    x = np.append(x, 1.0)
                np.testing.assert_allclose(result["pred"][t], x @ beta, rtol=rtol, atol=atol)


def fit(df, columns, **kwargs):
    return df.select(pds.rolling_lin_reg(*columns, target="y", **kwargs).alias("fit")).unnest("fit")


def test_closed_form_1d_matches_plugin(precision):
    parts = []
    for group in range(3):
        part = data(n=90, p=1, seed=group).with_columns(asset=pl.lit(group))
        part = part.with_columns(
            pl.when(pl.int_range(pl.len()) % 17 == 0).then(None).otherwise(pl.col("y")).alias("y"),
            pl.when(pl.int_range(pl.len()) % 29 == 0)
            .then(float("nan"))
            .otherwise(pl.col("x0"))
            .alias("x0"),
        )
        parts.append(part)
    df = pl.concat(parts)
    args = dict(window_size=31, half_life=8.5, min_valid_rows=12)
    plugin = df.select(
        pds.rolling_lin_reg("x0", target="y", add_bias=True, null_policy="skip", **args)
        .over("asset")
        .alias("fit")
    ).unnest("fit")
    native = df.select(
        pds.rolling_lin_reg_1d("x0", target="y", **args).over("asset").alias("fit")
    ).unnest("fit")
    assert_frame_equal(
        native,
        plugin,
        check_dtypes=False,
        rel_tol=2e-4 if not precision else 1e-8,
        abs_tol=2e-5 if not precision else 1e-10,
    )
    dtype = pl.Float64 if precision else pl.Float32
    assert native.schema == {"coeffs": pl.List(dtype), "pred": dtype}


def test_one_predictor_ewls_routes_to_closed_form(precision):
    df = data(n=90, p=1).with_columns(
        pl.when(pl.int_range(pl.len()) % 13 == 0).then(None).otherwise(pl.col("y")).alias("y")
    )
    args = dict(window_size=20, half_life=6, min_valid_rows=8)
    routed = df.select(
        pds.rolling_lin_reg("x0", target="y", add_bias=True, null_policy="skip", **args).alias(
            "fit"
        )
    )
    explicit = df.select(pds.rolling_lin_reg_1d("x0", target="y", **args).alias("fit"))
    assert_frame_equal(routed, explicit, check_exact=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 1, "half_life": 2},
        {"window_size": 4, "half_life": 0},
        {"window_size": 4, "half_life": float("nan")},
        {"window_size": 4, "half_life": 2, "min_valid_rows": 0},
        {"window_size": 4, "half_life": 2, "min_valid_rows": 5},
    ],
)
def test_closed_form_1d_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        pds.rolling_lin_reg_1d("x", target="y", **kwargs)


@pytest.mark.parametrize("p,bias", [(1, False), (1, True), (3, False), (3, True)])
@pytest.mark.parametrize("missing", ["none", "random", "block"])
def test_reference(precision, p, bias, missing):
    df = data(p=p)
    columns = [f"x{i}" for i in range(p)]
    if missing != "none":
        rng = np.random.default_rng(8)
        for c in [*columns, "y"]:
            mask = (
                rng.random(len(df)) < 0.08 if missing == "random" else np.arange(len(df)) % 80 < 30
            )
            df = df.with_columns(pl.when(pl.Series(mask)).then(None).otherwise(pl.col(c)).alias(c))
        df = df.with_columns(
            pl.when(pl.int_range(pl.len()) == 91)
            .then(float("nan"))
            .otherwise(pl.col("y"))
            .alias("y")
        )
    window, half_life, min_rows = 31, 8.5, p + int(bias) + 2
    result = fit(
        df,
        columns,
        window_size=window,
        half_life=half_life,
        add_bias=bias,
        min_valid_rows=min_rows,
        null_policy="skip",
    )
    assert_reference(df, result, columns, window, half_life, bias, min_rows)
    expected_dtype = pl.Float64 if precision else pl.Float32
    assert result.schema == {"coeffs": pl.List(expected_dtype), "pred": expected_dtype}


def test_missing_row_keeps_its_age(precision):
    df = pl.DataFrame({"x": [0.0, None, 2.0, 3.0], "y": [1.0, None, 0.0, 7.0]})
    result = fit(
        df, ["x"], window_size=4, half_life=1, add_bias=True, min_valid_rows=2, null_policy="skip"
    )
    x = np.array([[0, 1], [2, 1], [3, 1]], dtype=float)
    y = np.array([1, 0, 7], dtype=float)
    weights = np.array([1 / 8, 1 / 2, 1.0])  # Ages 3, 1, 0, not 2, 1, 0.
    expected = np.linalg.lstsq(x * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)[0]
    np.testing.assert_allclose(result["coeffs"][3], expected, rtol=1e-6 if precision else 2e-5)
    compressed = np.array([1 / 4, 1 / 2, 1.0])
    wrong = np.linalg.lstsq(x * np.sqrt(compressed)[:, None], y * np.sqrt(compressed), rcond=None)[
        0
    ]
    assert not np.allclose(expected, wrong)
    assert result["coeffs"][:3].null_count() == 3


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("column", ["x0", "y"])
def test_nonfinite_and_prediction(precision, value, column):
    df = data(n=35, p=1).with_columns(
        pl.when(pl.int_range(pl.len()) == 25).then(value).otherwise(pl.col(column)).alias(column)
    )
    result = fit(
        df, ["x0"], window_size=10, half_life=3, add_bias=True, min_valid_rows=3, null_policy="skip"
    )
    assert_reference(df, result, ["x0"], 10, 3, True, 3)
    assert result["coeffs"][25] is not None
    assert (result["pred"][25] is None) == (column == "x0")
    with pytest.raises(pl.exceptions.ComputeError, match="finite"):
        fit(df, ["x0"], window_size=10, half_life=3, null_policy="raise")


@pytest.mark.parametrize("n", [0, 1, 7, 8, 20])
def test_short_and_empty_inputs(precision, n):
    df = data(n=n, p=1)
    result = fit(df, ["x0"], window_size=8, half_life=2, add_bias=True)
    assert len(result) == n
    assert result["coeffs"][: min(n, 7)].null_count() == min(n, 7)


def test_all_null_and_minimum_count(precision):
    df = pl.DataFrame({"x": [None] * 20, "y": [None] * 20})
    result = fit(df, ["x"], window_size=8, half_life=2, add_bias=True, null_policy="skip")
    assert result["coeffs"].null_count() == len(df)
    df = pl.DataFrame(
        {"x": [None, 1.0, None, 2.0, 3.0, 4.0], "y": [None, 2.0, None, 5.0, 7.0, 8.0]}
    )
    result = fit(
        df, ["x"], window_size=4, half_life=2, add_bias=True, min_valid_rows=3, null_policy="skip"
    )
    assert result["coeffs"].is_null().to_list() == [True, True, True, True, False, False]
    # Two features plus intercept cannot be estimated from a two-row window.
    result = fit(data(n=10, p=2), ["x0", "x1"], window_size=2, half_life=2, add_bias=True)
    assert result["coeffs"].null_count() == 10


def test_intercept_only_and_no_coefficients(precision):
    df = pl.DataFrame({"y": [1.0, None, 4.0, 8.0]})
    result = fit(df, [], window_size=4, half_life=1, add_bias=True, null_policy="skip")
    expected = (1.0 / 8 + 4.0 / 2 + 8.0) / (1.0 / 8 + 1.0 / 2 + 1.0)
    np.testing.assert_allclose(result["coeffs"][3], [expected], rtol=1e-6)
    np.testing.assert_allclose(result["pred"][3], expected, rtol=1e-6)
    with pytest.raises(ValueError, match="predictor or an intercept"):
        pds.rolling_lin_reg(target="y", window_size=4, half_life=1)


def test_rank_deficiency_and_recovery(precision):
    df = data(n=180, p=2).with_columns(
        pl.when(pl.int_range(pl.len()).is_between(40, 119))
        .then(2 * pl.col("x0"))
        .otherwise(pl.col("x1"))
        .alias("x1")
    )
    result = fit(df, ["x0", "x1"], window_size=20, half_life=8, add_bias=True)
    assert result["coeffs"][59:120].null_count() == 61
    assert result["coeffs"][140:].null_count() == 0
    assert_reference(df, result, ["x0", "x1"], 20, 8, True, 3)


@pytest.mark.parametrize("column", ["x0", "y"])
@pytest.mark.parametrize("magnitude", [1e4, 1e8, 1e12])
def test_expiry_and_causality(precision, column, magnitude):
    df = data(n=140, p=1)
    args = dict(window_size=20, half_life=8, add_bias=True)
    original = fit(df, ["x0"], **args)
    changed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(magnitude).otherwise(pl.col(column)).alias(column)
    )
    after = fit(changed, ["x0"], **args)
    # Once row 0 expires, even a large outlier must leave no material trace.
    assert_frame_equal(
        after.slice(20),
        original.slice(20),
        rel_tol=2e-5 if not precision else 1e-8,
        abs_tol=2e-6 if not precision else 1e-10,
    )
    future = df.with_columns(
        pl.when(pl.int_range(pl.len()) >= 90).then(1e12).otherwise(pl.col(column)).alias(column)
    )
    assert_frame_equal(fit(future, ["x0"], **args).head(90), original.head(90))


def test_group_lazy_slice_and_chunks(precision):
    parts = [
        data(n=n, p=2, seed=i).with_columns(asset=pl.lit(str(i)))
        for i, n in enumerate([90, 7, 110])
    ]
    panel = pl.concat(parts, rechunk=False).slice(3)
    assert panel["x0"].n_chunks() > 1
    args = dict(window_size=15, half_life=5, add_bias=True, null_policy="skip")
    expr = pds.rolling_lin_reg("x0", "x1", target="y", **args).over("asset").alias("fit")
    got = panel.with_columns(expr)
    assert_frame_equal(got, panel.rechunk().with_columns(expr))
    assert_frame_equal(got, panel.lazy().with_columns(expr).collect())
    for sub in panel.partition_by("asset", maintain_order=True):
        expected = fit(sub, ["x0", "x1"], **args)
        actual = got.filter(pl.col("asset") == sub["asset"][0]).select("fit").unnest("fit")
        assert_frame_equal(actual, expected)
    # A sliced input starts its own window history.
    sliced = data(n=120, p=2).slice(11, 80)
    assert_reference(sliced, fit(sliced, ["x0", "x1"], **args), ["x0", "x1"], 15, 5, True, 2)
    # Slicing the output must retain the earlier rows used by the regression.
    whole = data(n=120, p=2)
    expr = pds.rolling_lin_reg("x0", "x1", target="y", **args).alias("fit")
    after = whole.lazy().select(expr).slice(11, 80).collect().unnest("fit")
    assert_frame_equal(after, fit(whole, ["x0", "x1"], **args).slice(11, 80))


def test_long_history(precision):
    df = data(n=50_000, p=3)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) % 17 == 0).then(None).otherwise(pl.col("y")).alias("y")
    )
    result = fit(
        df,
        ["x0", "x1", "x2"],
        window_size=504,
        half_life=126,
        min_valid_rows=126,
        add_bias=True,
        null_policy="skip",
    )
    indices = sorted(set([503, 504, 1007, 1008, 49_999, *range(997, 50_000, 997)]))
    assert_reference(df, result, ["x0", "x1", "x2"], 504, 126, True, 126, indices)


def test_against_weighted_lin_reg(precision):
    df = data(n=80, p=2).with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("y")).alias("y")
    )
    result = fit(df, ["x0", "x1"], window_size=20, half_life=6, add_bias=True, null_policy="skip")
    for t in [19, 40, 79]:
        sub = df.slice(t - 19, 20).with_columns(w=pl.Series(np.exp2(-np.arange(19, -1, -1) / 6)))
        sub = sub.drop_nulls()
        expected = sub.select(
            pds.lin_reg("x0", "x1", target="y", weights="w", add_bias=True)
        ).item()
        np.testing.assert_allclose(
            result["coeffs"][t],
            expected,
            rtol=2e-5 if not precision else 1e-8,
            atol=2e-6 if not precision else 1e-10,
        )


@pytest.mark.parametrize("half_life", [1e-10, 1e30])
def test_extreme_half_life(precision, half_life):
    df = data(n=60, p=1)
    result = fit(df, ["x0"], window_size=10, half_life=half_life)
    assert_reference(df, result, ["x0"], 10, half_life, False, 1)


def test_long_missing_block_preserves_fit_until_expiry(precision):
    window = 1500
    df = pl.DataFrame({"x": [1.0] * 3100, "y": [3.0] * window + [None] * window + [6.0] * 100})
    args = dict(window_size=window, half_life=1.1, null_policy="skip")
    result = fit(df, ["x"], **args)
    # Any nonempty window containing only y=3 has slope 3, even when all
    # unnormalized weights underflow. The last such observation expires at 2999.
    np.testing.assert_allclose(result["pred"][1499:2999], 3.0, rtol=2e-5)
    assert result["coeffs"][2999] is None
    assert result["pred"][2999] is None
    np.testing.assert_allclose(result["pred"][3000:], 6.0, rtol=2e-5)
    for t in [2670, 2680, 2684, 2998, 2999, 3000]:
        fresh = fit(df.slice(t - window + 1, window), ["x"], **args).tail(1)
        assert_frame_equal(result.slice(t, 1), fresh, rel_tol=2e-5, abs_tol=2e-6)


def test_missing_gap_decay_and_expiry(precision):
    df = data(n=125, p=2).with_columns(
        pl.when(
            pl.int_range(pl.len()).is_between(20, 64) | pl.int_range(pl.len()).is_between(78, 102)
        )
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    result = fit(
        df,
        ["x0", "x1"],
        window_size=40,
        half_life=2.5,
        min_valid_rows=5,
        add_bias=True,
        null_policy="skip",
    )
    assert_reference(df, result, ["x0", "x1"], 40, 2.5, True, 5)


@pytest.mark.parametrize("null_policy", ["SKIP", "RAISE"])
def test_null_policy_case_insensitive(precision, null_policy):
    df = data(n=25, p=1)
    args = dict(window_size=8, half_life=2)
    assert_frame_equal(
        fit(df, ["x0"], null_policy=null_policy, **args),
        fit(df, ["x0"], null_policy=null_policy.lower(), **args),
    )


@pytest.mark.parametrize("half_life", [1e-10, 0.001])
def test_recovery_after_overflow_and_decay(monkeypatch, half_life):
    monkeypatch.setattr(cfg, "LIN_REG_EXPR_F64", True)
    df = pl.DataFrame({"x": [1.0, 1.0, 1.0, 1e200, 2.0, 3.0], "y": [1.0, 1.0, 1.0, 2.0, 4.0, 6.0]})
    result = fit(df, ["x"], window_size=4, half_life=half_life)
    # The huge row's initial cross-product overflows, but its aged contribution
    # is finite (or zero). Recover as soon as the current window is estimable.
    assert_reference(df, result, ["x"], 4, half_life, False, 1, indices=[4, 5])


@pytest.mark.parametrize("l2_reg", [0.0, 0.1])
def test_none_preserves_legacy(precision, l2_reg):
    df = data(n=80, p=2)
    kwargs = dict(window_size=15, l2_reg=l2_reg, add_bias=True)
    assert_frame_equal(
        fit(df, ["x0", "x1"], **kwargs),
        fit(df, ["x0", "x1"], half_life=None, **kwargs),
        check_exact=True,
    )
    # Recursive regression shares the kwargs struct but has no half_life field.
    out = df.select(pds.recursive_lin_reg("x0", "x1", target="y", start_with=15))
    assert len(out) == len(df)


@pytest.mark.parametrize("half_life", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_half_life(half_life):
    with pytest.raises(ValueError, match="half_life"):
        pds.rolling_lin_reg("x", target="y", window_size=10, half_life=half_life)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"l2_reg": 0.1},
        {"l2_reg": -0.1},
        {"l2_reg": float("nan")},
        {"null_policy": "zero"},
        {"null_policy": "one"},
        {"null_policy": "ignore"},
        {"null_policy": "0.5"},
        {"min_valid_rows": 0},
        {"min_valid_rows": -1},
        {"min_valid_rows": 11},
    ],
)
def test_unsupported_parameters(kwargs):
    with pytest.raises(ValueError):
        pds.rolling_lin_reg("x", target="y", window_size=10, half_life=3, **kwargs)
