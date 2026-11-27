"""Quantile regression checked against an independent pinball-loss linear program."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack

import polars_ds as pds


def _objective(x, y, beta, quantile):
    residual = y - x @ beta
    return np.maximum(quantile * residual, (quantile - 1) * residual).mean()


def _lp(x, y, quantile):
    n, p = x.shape
    # X beta + positive_residual - negative_residual = y; beta is unrestricted.
    solution = linprog(
        np.r_[np.zeros(p), np.full(n, quantile), np.full(n, 1 - quantile)],
        A_eq=hstack([csc_matrix(x), eye(n), -eye(n)], format="csc"),
        b_eq=y,
        bounds=[(None, None)] * p + [(0, None)] * (2 * n),
        method="highs",
    )
    assert solution.success, solution.message
    return solution.x[:p], solution.fun / n


def _data(seed=390, n=80):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = x @ [0.8, -1.3] + 1.7 + rng.standard_t(4, size=n)
    return x, y, pl.from_numpy(x, schema=["x1", "x2"]).with_columns(y=pl.Series(y))


@pytest.mark.parametrize("quantile", [0.01, 0.1, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("bias", [False, True])
def test_quantile_objective_matches_independent_lp(quantile, bias):
    x, y, df = _data()
    design = np.c_[x, np.ones(len(y))] if bias else x
    _, optimal = _lp(design, y, quantile)
    beta = np.array(
        df.select(
            pds.quantile_reg(
                "x1",
                "x2",
                target="y",
                quantile=quantile,
                add_bias=bias,
                tol=1e-7,
                max_iter=20_000,
            )
        ).item()
    )
    assert len(beta) == design.shape[1]
    assert _objective(design, y, beta, quantile) <= optimal + 2e-6


@pytest.mark.parametrize("quantile", [0.25, 0.75])
def test_coefficients_match_statsmodels(quantile):
    from statsmodels.regression.quantile_regression import QuantReg

    x, y, df = _data(seed=21, n=120)
    design = np.c_[x, np.ones(len(y))]
    expected = QuantReg(y, design).fit(q=quantile, max_iter=20_000, p_tol=1e-9).params
    actual = np.array(
        df.select(
            pds.quantile_reg(
                "x1",
                "x2",
                target="y",
                quantile=quantile,
                add_bias=True,
                tol=1e-8,
                max_iter=20_000,
            )
        ).item()
    )
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("quantile", [0.25, 0.5])
def test_nonunique_intercept_fit_matches_objective(quantile):
    y = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.ones((len(y), 1))
    _, optimal = _lp(x, y, quantile)
    beta = np.array(
        pl.DataFrame({"y": y})
        .select(pds.quantile_reg(target="y", add_bias=True, quantile=quantile))
        .item()
    )
    assert _objective(x, y, beta, quantile) == pytest.approx(optimal, abs=1e-6)


def test_prediction_skip_preserves_original_row_alignment():
    df = pl.DataFrame(
        {
            "x": [0.0, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "y": [1.0, 99.0, 5.0, 7.0, None, 11.0, 13.0, 15.0],
        }
    )
    expr = pds.quantile_reg("x", target="y", add_bias=True, return_pred=True)
    actual = df.select(expr).unnest("quantile_pred")
    assert actual.schema == {"pred": pl.Float64, "resid": pl.Float64}
    assert actual["pred"].to_list() == pytest.approx(
        [1.0, None, 5.0, 7.0, None, 11.0, 13.0, 15.0],
        abs=1e-6,
    )
    assert actual["resid"].to_list() == pytest.approx(
        [0.0, None, 0.0, 0.0, None, 0.0, 0.0, 0.0],
        abs=1e-6,
    )
    assert_frame_equal(actual, df.lazy().select(expr).collect().unnest("quantile_pred"))


@pytest.mark.parametrize("policy,value", [("zero", 0.0), ("one", 1.0), ("1.25", 1.25)])
def test_fill_features_but_drop_null_targets(policy, value):
    df = pl.DataFrame({"x": [0.0, None, 2.0, 3.0, 4.0, 5.0], "y": [0.5, 1.1, 3.2, None, 3.9, 5.6]})
    clean = df.with_columns(pl.col("x").fill_null(value)).drop_nulls("y")
    x = np.c_[clean["x"].to_numpy(), np.ones(len(clean))]
    _, optimal = _lp(x, clean["y"].to_numpy(), 0.7)
    beta = np.array(
        df.select(
            pds.quantile_reg(
                "x",
                target="y",
                add_bias=True,
                quantile=0.7,
                null_policy=policy,
            )
        ).item()
    )
    assert _objective(x, clean["y"].to_numpy(), beta, 0.7) <= optimal + 1e-5
    result = df.select(
        pds.quantile_reg(
            "x",
            target="y",
            add_bias=True,
            quantile=0.7,
            null_policy=policy,
            return_pred=True,
        )
    ).unnest("quantile_pred")
    assert result["pred"].null_count() == 1
    assert result["pred"][3] is None
    assert result["pred"][1] == pytest.approx(value * beta[0] + beta[1], abs=1e-6)


def test_grouped_and_lazy_fits():
    frames = []
    for group in range(3):
        _, _, frame = _data(seed=390 + group, n=50)
        frames.append(frame.with_columns(group=pl.lit(group)))
    df = pl.concat(frames, rechunk=False)
    expr = pds.quantile_reg("x1", "x2", target="y", quantile=0.8, add_bias=True)
    eager = df.group_by("group").agg(expr).sort("group")
    lazy = df.lazy().group_by("group").agg(expr).sort("group").collect()
    assert_frame_equal(eager, lazy)
    for group, beta in eager.iter_rows():
        part = df.filter(pl.col("group") == group)
        design = np.c_[part.select("x1", "x2").to_numpy(), np.ones(len(part))]
        _, optimal = _lp(design, part["y"].to_numpy(), 0.8)
        assert _objective(design, part["y"].to_numpy(), np.array(beta), 0.8) <= optimal + 2e-5


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_feature_and_target_scaling(scale):
    _, _, df = _data()
    original = np.array(df.select(pds.quantile_reg("x1", "x2", target="y", add_bias=True)).item())
    scaled = df.with_columns(pl.col("x1") * scale, pl.col("y") * scale)
    actual = np.array(scaled.select(pds.quantile_reg("x1", "x2", target="y", add_bias=True)).item())
    np.testing.assert_allclose(actual, original * [1, scale, scale], rtol=1e-4, atol=1e-12)


@pytest.mark.parametrize(
    "name,value",
    [
        ("quantile", 0),
        ("quantile", 1),
        ("quantile", float("nan")),
        ("quantile", float("inf")),
        ("tol", 0),
        ("tol", -1),
        ("tol", float("nan")),
        ("tol", float("inf")),
        ("max_iter", 0),
        ("max_iter", -1),
        ("max_iter", 1.5),
    ],
)
def test_invalid_options(name, value):
    with pytest.raises(ValueError, match=name):
        pds.quantile_reg("x", target="y", **{name: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("column", ["x1", "y"])
def test_nonfinite_values_are_rejected(value, column):
    _, _, df = _data()
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 1).then(value).otherwise(pl.col(column)).alias(column)
    )
    with pytest.raises(pl.exceptions.ComputeError, match="finite"):
        df.select(pds.quantile_reg("x1", "x2", target="y"))


def test_rank_deficient_design_and_empty_input_are_rejected():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 5.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="rank"):
        df.select(pds.quantile_reg("x", "x", target="y"))
    with pytest.raises(pl.exceptions.ComputeError, match="empty|samples"):
        df.clear().select(pds.quantile_reg("x", target="y"))
    with pytest.raises(pl.exceptions.ComputeError, match="samples"):
        df.head(1).select(pds.quantile_reg("x", target="y", add_bias=True))
    with pytest.raises(pl.exceptions.ComputeError, match="empty|samples"):
        df.clear().select(pds.quantile_reg(pl.lit(1.0), target="y"))


def test_null_raise_ignore_and_invalid_policy():
    df = pl.DataFrame({"x": [1.0, None, 3.0, 4.0], "y": [2.0, 4.0, 5.0, 7.0]})
    for policy in ["raise", "ignore"]:
        with pytest.raises(pl.exceptions.ComputeError, match="null"):
            df.select(pds.quantile_reg("x", target="y", null_policy=policy))
    with pytest.raises(pl.exceptions.ComputeError, match="null_policy"):
        df.select(pds.quantile_reg("x", target="y", null_policy="bogus"))


def test_iteration_limit_is_an_error():
    _, _, df = _data()
    with pytest.raises(pl.exceptions.ComputeError, match="converge"):
        df.select(pds.quantile_reg("x1", "x2", target="y", quantile=0.9, tol=1e-12, max_iter=1))


def test_integer_float32_and_broadcast_inputs():
    df = pl.DataFrame({"x": [0, 1, 2, 3], "y": [1.0, 3.0, 5.0, 7.0]})
    expr = pds.quantile_reg(pl.col("x").cast(pl.Float32), target="y", add_bias=True)
    beta = df.select(expr).item()
    np.testing.assert_allclose(beta, [2.0, 1.0], atol=1e-6)
    # A literal predictor is broadcast to the target length, just like a bias column.
    actual = df.select(pds.quantile_reg(pl.lit(1.0), target="y")).item()
    # At q=0.5 pinball loss is half the mean absolute residual (2.0 / 2).
    assert _objective(np.ones((4, 1)), df["y"].to_numpy(), np.array(actual), 0.5) == pytest.approx(
        1.0
    )


def test_grouped_predictions_with_nulls():
    df = pl.DataFrame(
        {
            "group": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "x": [0.0, 0.0, None, 1.0, 2.0, 2.0, 3.0, 3.0],
            "y": [1.0, 2.0, 99.0, 5.0, 5.0, None, 7.0, 11.0],
        }
    )
    expr = pds.quantile_reg("x", target="y", add_bias=True, return_pred=True).over("group")
    out = df.select(expr).unnest("quantile_pred")
    assert out["pred"].to_list() == pytest.approx(
        [1.0, 2.0, None, 5.0, 5.0, None, 7.0, 11.0], abs=1e-6
    )
    assert_frame_equal(out, df.lazy().select(expr).collect().unnest("quantile_pred"))


def test_invalid_input_structure_and_all_null_rows():
    with pytest.raises(ValueError, match="predictor"):
        pds.quantile_reg(target="y")
    with pytest.raises(ValueError, match="single"):
        pds.quantile_reg("x", target=["a", "b"])
    df = pl.DataFrame({"x": [None, None], "y": [1.0, 2.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="empty|samples"):
        df.select(pds.quantile_reg("x", target="y"))
    with pytest.raises(pl.exceptions.ComputeError, match="finite"):
        df.select(pds.quantile_reg("x", target="y", null_policy="inf"))


@pytest.mark.parametrize("feature,target", [(1e308, 1e-308), (1e-308, 1e308)])
def test_unrepresentable_coefficients_raise(feature, target):
    df = pl.DataFrame({"x": [feature, feature / 2], "y": [target, target / 2]})
    with pytest.raises(pl.exceptions.ComputeError, match="Float64 range"):
        df.select(pds.quantile_reg("x", target="y"))


def test_zero_response_and_null_free_ignore():
    df = pl.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0], "y": [0.0] * 4})
    beta = df.select(pds.quantile_reg("x", target="y", add_bias=True, null_policy="ignore")).item()
    np.testing.assert_array_equal(beta, [0.0, 0.0])


def test_difficult_tail_fit_with_larger_iteration_budget():
    # Rounded responses and a tail quantile can require more than the default budget.
    rng = np.random.default_rng(39003)
    x = rng.normal(size=(60, 4))
    design = np.c_[x, np.ones(60)]
    y = np.round(design @ rng.normal(size=5) + rng.standard_t(3, size=60))
    columns = [f"x{i}" for i in range(4)]
    df = pl.from_numpy(x, schema=columns).with_columns(y=pl.Series(y))
    beta = np.array(
        df.select(
            pds.quantile_reg(*columns, target="y", quantile=0.01, add_bias=True, max_iter=100_000)
        ).item()
    )
    _, optimal = _lp(design, y, 0.01)
    loss = _objective(design, y, beta, 0.01)
    assert loss - optimal <= 1e-6 * (np.max(np.abs(y)) + loss) + 1e-8
