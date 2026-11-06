"""Analytic checks for rcond in design singular-value units."""

import numpy as np
import polars as pl
import pytest

import polars_ds as pds
import polars_ds.config as cfg


@pytest.fixture(params=[True, False], ids=["f64", "f32"])
def precision(request, monkeypatch):
    monkeypatch.setattr(cfg, "LIN_REG_EXPR_F64", request.param)
    return np.float64 if request.param else np.float32


@pytest.fixture(
    params=[{}, {"stable": False}, {"stable": True}], ids=["default", "normal", "stable"]
)
def solver_kwargs(request):
    return request.param


def fit(x, y, **kwargs):
    names = [f"x{i}" for i in range(x.shape[1])]
    frame = pl.DataFrame({**dict(zip(names, x.T)), "y": y})
    result = frame.select(pds.lin_reg_w_rcond(*names, target="y", **kwargs).alias("fit"))
    result = result.unnest("fit")
    return result["coeffs"][0].to_numpy(), result["singular_values"][0].to_numpy()


def test_small_exposure_is_not_truncated(precision, solver_kwargs):
    z = np.arange(-2.0, 3.0)
    x = (z * (1e-7 if precision == np.float64 else 1e-4))[:, None]
    y = 0.01 + 0.001 * z
    beta, spectrum = fit(x, y, add_bias=True, rcond=1e-12, **solver_kwargs)
    expected = np.linalg.lstsq(np.column_stack([x, np.ones(5)]), y, rcond=1e-12)[0]
    np.testing.assert_allclose(beta, expected, rtol=2e-5, atol=1e-6)
    assert np.all(spectrum > 0)


@pytest.mark.parametrize("scale", [0.125, 1.0, 128.0])
@pytest.mark.parametrize("rcond", [0.01, 0.125, 0.25, 1.0])
def test_cutoff_uses_design_units_and_excludes_boundary(precision, scale, rcond, solver_kwargs):
    singular = np.array([1.0, 0.125, 0.015625]) * scale
    x = np.vstack([np.diag(singular), np.zeros((3, 3))])
    beta = np.array([2.0, -3.0, 4.0])
    actual, spectrum = fit(x, x @ beta, rcond=rcond, **solver_kwargs)
    # Dyadic ratios give exact boundary cases at scale=1, independent of LAPACK.
    expected = np.where(singular > rcond * singular[0], beta, 0.0)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_allclose(spectrum, singular, rtol=2e-6)


def test_zero_design_returns_finite_zero_coefficients(precision, solver_kwargs):
    beta, spectrum = fit(np.zeros((8, 2)), np.ones(8), **solver_kwargs)
    np.testing.assert_array_equal(beta, [0.0, 0.0])
    np.testing.assert_array_equal(spectrum, [0.0, 0.0])


@pytest.mark.parametrize("l2", [0.0, 0.2])
@pytest.mark.parametrize("bias", [False, True])
def test_default_matches_normal_and_both_paths_match_lapack(precision, l2, bias):
    rng = np.random.default_rng(976)
    x = rng.normal(size=(65, 3)).astype(precision)
    y = (0.7 + x @ [0.2, -0.4, 0.6]).astype(precision)
    kwargs = dict(add_bias=bias, rcond=1e-6, l2_reg=l2)
    default = fit(x, y, **kwargs)
    normal = fit(x, y, stable=False, **kwargs)
    stable = fit(x, y, stable=True, **kwargs)
    design = np.column_stack([x, np.ones(len(x))]) if bias else x.astype(float)
    penalty = np.eye(design.shape[1])
    if bias:
        penalty[-1, -1] = 0.0
    augmented = np.vstack([design, np.sqrt(precision(l2)) * penalty])
    expected, _, _, singular = np.linalg.lstsq(
        augmented, np.r_[y, np.zeros(design.shape[1])], rcond=1e-6
    )
    for actual in [normal, stable]:
        np.testing.assert_allclose(actual[0], expected, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(actual[1], singular, rtol=2e-5, atol=1e-6)
        assert actual[0].dtype == actual[1].dtype == precision
    for actual, explicit in zip(default, normal):
        np.testing.assert_array_equal(actual, explicit)


@pytest.mark.parametrize("bias", [False, True])
def test_truncated_ridge_matches_augmented_design(precision, solver_kwargs, bias):
    x = np.vstack([np.diag([1.0, 0.125, 0.015625]), np.zeros((5, 3))])
    y = x @ [2.0, -3.0, 4.0] + 0.5
    beta, spectrum = fit(x, y, add_bias=bias, rcond=0.2, l2_reg=0.01, **solver_kwargs)
    design = np.column_stack([x, np.ones(len(x))]) if bias else x
    penalty = np.eye(design.shape[1])
    if bias:
        penalty[-1, -1] = 0.0
    augmented = np.vstack([design, np.sqrt(precision(0.01)) * penalty])
    expected, _, rank, singular = np.linalg.lstsq(
        augmented, np.r_[y.astype(precision), np.zeros(design.shape[1])], rcond=0.2
    )
    assert rank < design.shape[1], "must exercise Ridge truncation"
    np.testing.assert_allclose(beta, expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(spectrum, singular, rtol=2e-5, atol=1e-6)
