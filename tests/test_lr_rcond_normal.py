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


def fit(x, y, **kwargs):
    names = [f"x{i}" for i in range(x.shape[1])]
    frame = pl.DataFrame({**dict(zip(names, x.T)), "y": y})
    result = frame.select(pds.lin_reg_w_rcond(*names, target="y", **kwargs).alias("fit"))
    result = result.unnest("fit")
    return result["coeffs"][0].to_numpy(), result["singular_values"][0].to_numpy()


def test_small_exposure_is_not_truncated(precision):
    z = np.arange(-2.0, 3.0)
    x = (z * (1e-7 if precision == np.float64 else 1e-4))[:, None]
    y = 0.01 + 0.001 * z
    beta, spectrum = fit(x, y, add_bias=True, rcond=1e-12)
    expected = np.linalg.lstsq(np.column_stack([x, np.ones(5)]), y, rcond=1e-12)[0]
    np.testing.assert_allclose(beta, expected, rtol=2e-5, atol=1e-6)
    assert np.all(spectrum > 0)


@pytest.mark.parametrize("scale", [0.125, 1.0, 128.0])
@pytest.mark.parametrize("rcond", [0.01, 0.125, 0.25, 1.0])
def test_cutoff_uses_design_units_and_excludes_boundary(precision, scale, rcond):
    singular = np.array([1.0, 0.125, 0.015625]) * scale
    x = np.vstack([np.diag(singular), np.zeros((3, 3))])
    beta = np.array([2.0, -3.0, 4.0])
    actual, spectrum = fit(x, x @ beta, rcond=rcond)
    # Dyadic ratios give exact boundary cases at scale=1, independent of LAPACK.
    expected = np.where(singular > rcond * singular[0], beta, 0.0)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_allclose(spectrum, singular, rtol=2e-6)


def test_zero_design_returns_finite_zero_coefficients(precision):
    beta, spectrum = fit(np.zeros((8, 2)), np.ones(8))
    np.testing.assert_array_equal(beta, [0.0, 0.0])
    np.testing.assert_array_equal(spectrum, [0.0, 0.0])
