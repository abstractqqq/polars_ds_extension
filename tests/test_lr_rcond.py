"""The rcond cutoff acts on design singular values, not Gram eigenvalues."""

import numpy as np
import polars as pl
import pytest

import polars_ds as pds
import polars_ds.config as cfg


@pytest.fixture(params=[True, False], ids=["f64", "f32"])
def precision(request, monkeypatch):
    monkeypatch.setattr(cfg, "LIN_REG_EXPR_F64", request.param)
    return np.float64 if request.param else np.float32


def solve(x, y, *, bias=False, rcond=1e-12, l2=0.0, stable=True):
    columns = [f"x{i}" for i in range(x.shape[1])]
    frame = pl.DataFrame({**dict(zip(columns, x.T)), "y": y})
    result = (
        frame.lazy()
        .select(
            pds.lin_reg_w_rcond(
                *columns, target="y", add_bias=bias, rcond=rcond, l2_reg=l2, stable=stable
            ).alias("fit")
        )
        .unnest("fit")
        .collect()
    )
    return result["coeffs"][0].to_numpy(), result["singular_values"][0].to_numpy()


def reference(x, y, precision, *, bias=False, rcond=1e-12, l2=0.0):
    # Quantize exactly as the plugin, then use independent Float64 LAPACK solves.
    design = x.astype(precision).astype(float)
    target = y.astype(precision).astype(float)
    if bias:
        design = np.column_stack([design, np.ones(len(design))])
    effective = max(abs(rcond), np.finfo(precision).eps * max(design.shape))
    if l2 > 0:
        p = design.shape[1] - int(bias)
        penalty = np.zeros((p, design.shape[1]))
        penalty[np.arange(p), np.arange(p)] = np.sqrt(precision(l2))
        design = np.vstack([design, penalty])
        target = np.concatenate([target, np.zeros(p)])
    beta, _, rank, spectrum = np.linalg.lstsq(design, target, rcond=effective)
    return beta, spectrum, rank, effective


def assert_reference(beta, spectrum, expected, precision, *, tolerance=None):
    tolerance = tolerance or (
        dict(rtol=1e-8, atol=1e-10) if precision == np.float64 else dict(rtol=2e-4, atol=1e-5)
    )
    np.testing.assert_allclose(beta, expected[0], **tolerance)
    np.testing.assert_allclose(spectrum, expected[1], **tolerance)
    assert np.count_nonzero(spectrum > expected[3] * spectrum[0]) == expected[2]


def test_small_exposure_is_not_truncated(precision):
    z = np.arange(-2.0, 3.0)
    x = (z * (1e-7 if precision == np.float64 else 1e-4))[:, None]
    y = 0.01 + 0.001 * z
    beta, spectrum = solve(x, y, bias=True)
    expected = reference(x, y, precision, bias=True)
    assert expected[2] == 2
    assert_reference(beta, spectrum, expected, precision)


def test_float32_duplicate_features_with_ridge_preserve_symmetry(monkeypatch):
    monkeypatch.setattr(cfg, "LIN_REG_EXPR_F64", False)
    rng = np.random.default_rng(20261103)
    z = rng.normal(size=8193).astype(np.float32)
    y = rng.normal(size=len(z)).astype(np.float32)
    x = np.column_stack([z, z])
    beta, spectrum = solve(x, y, l2=0.2)
    # Identical penalized predictors must have identical coefficients. This
    # closed form is independent of both QR and LAPACK's treatment of rank.
    z64, y64 = z.astype(float), y.astype(float)
    expected = np.dot(z64, y64) / (2 * np.dot(z64, z64) + float(np.float32(0.2)))
    assert np.all(np.isfinite(spectrum))
    np.testing.assert_allclose(beta, [expected, expected], rtol=2e-4, atol=1e-5)


def test_qr_does_not_truncate_before_the_requested_svd_cutoff(monkeypatch):
    monkeypatch.setattr(cfg, "LIN_REG_EXPR_F64", True)
    rng = np.random.default_rng(20261103)
    # Generate the same prefix of a longer sample to make the failure stable.
    z = rng.normal(size=8193)[:4096]
    w = rng.normal(size=8193)[:4096]
    x = np.column_stack([z, z + 1e-11 * w])
    expected = np.array([2.0, -3.0])
    beta, spectrum = solve(x, x @ expected, rcond=1e-12)
    cutoff = max(1e-12, np.finfo(float).eps * max(x.shape))
    # Both directions survive the public cutoff, so the exact noiseless model
    # must be recovered, even if the QR backend considers a column negligible.
    assert spectrum[-1] > cutoff * spectrum[0]
    np.testing.assert_allclose(beta, expected, rtol=1e-3, atol=1e-3)


def test_float32_cutoff_agrees_with_the_returned_rounded_spectrum(monkeypatch):
    monkeypatch.setattr(cfg, "LIN_REG_EXPR_F64", False)
    rng = np.random.default_rng(1)
    u = np.linalg.qr(rng.normal(size=(16, 2)))[0]
    v = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    x = (u * [1.0, 0.1]) @ v.T
    y = u[:, 1]
    full_beta, spectrum = solve(x, y)
    assert np.linalg.norm(full_beta) > 1.0
    rcond = np.float32(spectrum[1] / spectrum[0])
    beta, spectrum = solve(x, y, rcond=float(rcond))
    assert spectrum[1] <= rcond * spectrum[0]
    # The response points along the discarded direction. A solve using hidden
    # Float64 singular values used to keep a coefficient vector of norm ten.
    assert np.linalg.norm(beta) < 1e-5


@pytest.mark.parametrize("scale_index", [-1, 0, 1])
def test_common_rescaling_preserves_coefficients_and_rank(precision, scale_index):
    z = np.arange(-2.0, 3.0)
    small = 1e-7 if precision == np.float64 else 1e-4
    exponent = 120 if precision == np.float64 else 12
    scale = 10.0 ** (exponent * scale_index)
    x = np.column_stack([z * small, np.ones(5)]) * scale
    y = (0.01 + 0.001 * z) * scale
    beta, spectrum = solve(x, y)
    expected = reference(x, y, precision)
    assert expected[2] == 2
    # Scale the spectra back so an absolute tolerance cannot hide tiny values.
    assert_reference(
        beta, spectrum / scale, (expected[0], expected[1] / scale, *expected[2:]), precision
    )


@pytest.mark.parametrize("rcond", [0.01, 0.125, 0.25, 1.0])
def test_cutoff_uses_original_spectrum_and_excludes_the_boundary(precision, rcond):
    singular = np.array([1.0, 0.125, 0.015625])
    x = np.vstack([np.diag(singular), np.zeros((3, 3))])
    beta = np.array([2.0, -3.0, 4.0])
    actual, spectrum = solve(x, x @ beta, rcond=rcond)
    # Analytic diagonal design also defines the exact-boundary behavior, without
    # depending on LAPACK's special handling of rcond >= 1.
    np.testing.assert_allclose(actual, np.where(singular > rcond, beta, 0.0), atol=1e-6)
    np.testing.assert_allclose(spectrum, singular, rtol=1e-6)


@pytest.mark.parametrize("stable", [False, True])
def test_zero_rcond_keeps_the_documented_machine_precision_floor(precision, stable):
    x = np.zeros((64, 2))
    x[0, 0], x[1, 1] = 1.0, 1e-15 if precision == np.float64 else 1e-6
    y = x @ np.array([2.0, 3.0])
    beta, spectrum = solve(x, y, rcond=0.0, stable=stable)
    expected = reference(x, y, precision, rcond=0.0)
    assert expected[2] == 1
    assert_reference(beta, spectrum, expected, precision)


@pytest.mark.parametrize("kind", ["zero", "duplicate", "constant"])
@pytest.mark.parametrize("bias", [False, True])
def test_rank_deficient_design_returns_finite_minimum_norm_solution(precision, kind, bias):
    z = np.linspace(-1.0, 1.0, 16)
    x = {
        "zero": np.zeros((16, 2)),
        "duplicate": np.column_stack([z, z]),
        "constant": np.ones((16, 2)),
    }[kind]
    y = 0.5 + 2.0 * z
    beta, spectrum = solve(x, y, bias=bias)
    assert np.isfinite(beta).all()
    assert_reference(beta, spectrum, reference(x, y, precision, bias=bias), precision)


def test_near_collinear_full_rank_design_keeps_the_resolvable_direction(precision):
    rng = np.random.default_rng(20261103)
    q, _ = np.linalg.qr(rng.normal(size=(512, 2)))
    v = np.array([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    small = 1e-9 if precision == np.float64 else 1e-3
    x = (q * [1.0, small]) @ v.T
    y = q @ [1.0, 0.1]
    beta, spectrum = solve(x, y)
    expected = reference(x, y, precision)
    assert expected[2] == 2
    tolerance = 2e-6 if precision == np.float64 else 2e-3
    np.testing.assert_allclose(beta, expected[0], rtol=tolerance, atol=1e-8)
    np.testing.assert_allclose(spectrum, expected[1], rtol=tolerance, atol=1e-14)
    quantized = x.astype(precision).astype(float)
    assert np.linalg.norm(quantized @ beta - y) / np.linalg.norm(y) < tolerance


@pytest.mark.parametrize("l2", [1e-6, 0.2, 5.0])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rcond", [1e-12, 0.2])
def test_ridge_matches_augmented_design_solution_and_spectrum(precision, l2, bias, rcond):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 2))
    x[:, 0] *= 1e-7 if precision == np.float64 else 1e-4
    y = 0.7 + x @ [0.2, -0.4]
    beta, spectrum = solve(x, y, bias=bias, l2=l2, rcond=rcond)
    assert_reference(
        beta, spectrum, reference(x, y, precision, bias=bias, l2=l2, rcond=rcond), precision
    )


def test_ridge_does_not_penalize_the_intercept_in_a_rank_deficient_design(precision):
    x, y = np.ones((12, 1)), np.full(12, 2.0)
    beta, spectrum = solve(x, y, bias=True, l2=3.0)
    np.testing.assert_allclose(beta, [0.0, 2.0], atol=1e-5)
    assert_reference(beta, spectrum, reference(x, y, precision, bias=True, l2=3.0), precision)


@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize("engine", ["auto", "streaming", "in-memory"])
def test_grouped_sliced_multichunk_inputs_skip_null_rows(precision, engine, stable):
    z = np.tile(np.arange(-10.0, 10.0), 3)
    small = 1e-7 if precision == np.float64 else 1e-4
    x, y = (small * z).tolist(), (0.01 + 0.001 * z).tolist()
    x[7], y[25] = None, None
    frame = pl.DataFrame({"x": x, "y": y, "g": np.repeat(np.arange(3), 20)})
    frame = pl.concat([frame[:17], frame[17:]], rechunk=False).slice(2)
    result = (
        frame.lazy()
        .group_by("g")
        .agg(
            pds.lin_reg_w_rcond(
                "x", target="y", add_bias=True, rcond=1e-12, null_policy="skip", stable=stable
            )
            .first()
            .alias("fit")
        )
        .unnest("fit")
        .sort("g")
        .collect(engine=engine)
    )
    dtype = pl.Float64 if precision == np.float64 else pl.Float32
    assert result.schema == {
        "g": pl.Int64,
        "coeffs": pl.List(dtype),
        "singular_values": pl.List(dtype),
    }
    for group, part in enumerate(frame.partition_by("g", maintain_order=True)):
        xx, yy = part["x"].to_numpy(), part["y"].to_numpy()
        valid = np.isfinite(xx) & np.isfinite(yy)
        expected = reference(xx[valid, None], yy[valid], precision, bias=True)
        # The normal-equations Float32 path retains its original arithmetic:
        # the scaled predictor amplifies roundoff beyond the stable-path bound.
        tolerance = dict(rtol=1e-3, atol=1e-5) if precision == np.float32 and not stable else None
        beta = result["coeffs"][group].to_numpy()
        assert_reference(
            beta,
            result["singular_values"][group].to_numpy(),
            expected,
            precision,
            tolerance=tolerance,
        )
        design = np.column_stack([xx[valid].astype(precision), np.ones(valid.sum())])
        prediction_error = np.linalg.norm(design @ (beta - expected[0])) / np.linalg.norm(
            design @ expected[0]
        )
        assert prediction_error < (1e-8 if precision == np.float64 else 1e-3)


@pytest.mark.parametrize("rows", [512, 513, 4096, 4097, 1_000_000])
@pytest.mark.parametrize("l2", [0.0, 0.2])
def test_tall_design_and_qr_block_boundaries(precision, rows, l2):
    # The pinned faer's unbounded Float32 QR collapses the million-row design
    # to rank one. Also check exact/partial block ends and propagation of Ridge.
    rng = np.random.default_rng(20261103)
    x = rng.normal(size=(rows, 3))
    y = 0.7 + x @ [0.3, 0.6, 1.0]
    beta, spectrum = solve(x, y, bias=True, l2=l2)
    expected = reference(x, y, precision, bias=True, l2=l2)
    assert expected[2] == 4
    assert_reference(beta, spectrum, expected, precision)


@pytest.mark.parametrize("direction", [-1, 1])
def test_extreme_global_scaling_preserves_the_design_spectrum(precision, direction):
    rng = np.random.default_rng(912)
    exponent = 280 if precision == np.float64 else 30
    scale = 10.0 ** (direction * exponent)
    x = rng.normal(size=(257, 5)) * scale
    y = rng.normal(size=257) * scale
    beta, spectrum = solve(x, y)
    expected = reference(x, y, precision)
    assert_reference(
        beta, spectrum / scale, (expected[0], expected[1] / scale, *expected[2:]), precision
    )


@pytest.mark.parametrize("prefix", [0.0, 1e-280])
def test_later_blocks_can_supply_rank_after_a_zero_or_tiny_prefix(precision, prefix):
    rng = np.random.default_rng(301)
    x = rng.normal(size=(8193, 3))
    x[:4096] *= prefix
    y = x @ np.array([1.5, -2.0, 0.75])
    beta, spectrum = solve(x, y)
    assert_reference(beta, spectrum, reference(x, y, precision), precision)


@pytest.mark.parametrize("features", [1, 20, 65, 129])
@pytest.mark.parametrize("l2", [0.0, 0.2])
def test_feature_permutations_restore_coefficient_order(precision, features, l2):
    rng = np.random.default_rng(702)
    x = rng.normal(size=(3 * features + 1, features))
    x *= np.linspace(0.2, 2.0, features)
    y = rng.normal(size=len(x))
    order = rng.permutation(features)
    beta, spectrum = solve(x[:, order], y, l2=l2)
    assert_reference(beta, spectrum, reference(x[:, order], y, precision, l2=l2), precision)


@pytest.mark.parametrize("column", ["x", "y"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_raise_after_null_handling(precision, column, value):
    values = {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}
    values[column][1] = value
    with pytest.raises(pl.exceptions.ComputeError, match="must be finite after null handling"):
        pl.DataFrame(values).lazy().select(
            pds.lin_reg_w_rcond("x", target="y", null_policy="ignore", stable=True)
        ).collect()


@pytest.mark.parametrize("gap", [600, 900, 1000])
def test_rescaling_does_not_overflow_when_the_coefficient_is_representable(monkeypatch, gap):
    monkeypatch.setattr(cfg, "LIN_REG_EXPR_F64", True)
    # The normalized solution is 2**-gap and the scale ratio is 2**1600.
    # Forming the ratio would overflow; an extra QR scale left in the SVD can
    # instead underflow the projection before coefficient rescaling recovers it.
    x = np.array([[np.ldexp(1.0, -600)], [0.0]])
    y = np.array([np.ldexp(1.0, 1000 - gap), np.ldexp(1.0, 1000)])
    beta, spectrum = solve(x, y)
    np.testing.assert_allclose(beta / np.ldexp(1.0, 1600 - gap), [1.0], rtol=1e-12)
    np.testing.assert_allclose(spectrum / np.ldexp(1.0, -600), [1.0], rtol=1e-12)


def test_zero_target_produces_zero_coefficients_without_rescaling_by_zero(precision):
    rng = np.random.default_rng(502)
    x = rng.normal(size=(20, 3))
    y = np.zeros(len(x))
    beta, spectrum = solve(x, y, bias=True, l2=0.2)
    assert_reference(beta, spectrum, reference(x, y, precision, bias=True, l2=0.2), precision)


@pytest.mark.parametrize("l2", [0.0, 0.2, 1e30])
def test_intercept_only_is_independent_of_ridge_penalty(precision, l2):
    y = np.linspace(-1.0, 4.0, 16)
    beta, spectrum = solve(np.empty((len(y), 0)), y, bias=True, l2=l2)
    np.testing.assert_allclose(beta, [np.mean(y.astype(precision))], rtol=1e-6)
    np.testing.assert_allclose(spectrum, [np.sqrt(len(y))], rtol=1e-6)


@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize("rows", [513, 4097])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("l2", [0.0, 0.2])
def test_untruncated_fit_satisfies_first_order_optimality(
    precision, rows, bias, l2, stable, record_property
):
    rng = np.random.default_rng(915)
    x = rng.normal(size=(rows, 5)).astype(precision).astype(float)
    y = (0.7 + x @ np.arange(1.0, 6.0) + rng.normal(size=rows)).astype(precision).astype(float)
    beta, spectrum = solve(x, y, bias=bias, l2=l2, stable=stable)
    design = np.column_stack([x, np.ones(rows)]) if bias else x
    cutoff = max(1e-12, np.finfo(precision).eps * max(design.shape))
    assert np.all(spectrum > cutoff * spectrum[0]), "optimality requires no truncation"

    residual = design @ beta - y
    penalty = np.ones(len(beta))
    if bias:
        penalty[-1] = 0.0
    gradient = design.T @ residual + float(precision(l2)) * penalty * beta
    # A dimensionless stationarity residual, evaluated in Float64 even when
    # coefficients are returned in Float32. For OLS this tests orthogonality;
    # for Ridge it tests the gradient, including the unpenalized intercept.
    denominator = np.linalg.norm(design, 2) * (
        np.linalg.norm(design, 2) * np.linalg.norm(beta) + np.linalg.norm(y)
    ) + float(precision(l2)) * np.linalg.norm(penalty * beta)
    error = np.linalg.norm(gradient) / denominator
    record_property("stationarity_error", float(error))
    # Normal-equations Float32 also rounds the Gram product and its SVD;
    # allow eight eps rather than the stable path's output-rounding bound.
    limit = 2e-14 if precision == np.float64 else (2e-7 if stable else 8 * np.finfo(precision).eps)
    assert error < limit, error


@pytest.mark.parametrize("l2", [0.0, 1e-8, 0.2])
def test_ill_conditioned_fit_checks_coefficients_predictions_and_objective(
    precision, l2, record_property
):
    rng = np.random.default_rng(812)
    q, _ = np.linalg.qr(rng.normal(size=(513, 3)))
    v, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    # The weakest direction is above the configured cutoff in both precisions.
    weak = 1e-7 if precision == np.float64 else 2e-3
    x = ((q * [1.0, 0.1, weak]) @ v.T).astype(precision).astype(float)
    y = (q @ [1.0, -0.3, 0.02] + 0.01 * rng.normal(size=len(q))).astype(precision).astype(float)
    beta, spectrum = solve(x, y, l2=l2)
    expected, _, rank, cutoff = reference(x, y, precision, l2=l2)
    assert rank == 3
    assert np.count_nonzero(spectrum > cutoff * spectrum[0]) == rank
    lam = float(precision(l2))

    def objective(coeffs):
        residual = x @ coeffs - y
        return residual @ residual + lam * (coeffs @ coeffs)

    errors = {
        "coefficients": np.linalg.norm(beta - expected) / np.linalg.norm(expected),
        "predictions": np.linalg.norm(x @ beta - x @ expected) / np.linalg.norm(x @ expected),
        "objective": abs(objective(beta) - objective(expected)) / objective(expected),
    }
    # Separate bounds matter: an accurate prediction/objective can conceal a
    # poor coefficient vector in a weakly identified direction.
    limits = (
        {"coefficients": 2e-7, "predictions": 2e-9, "objective": 2e-11}
        if precision == np.float64
        else {"coefficients": 2e-4, "predictions": 2e-5, "objective": 2e-6}
    )
    for metric, limit in limits.items():
        record_property(metric + "_relative_error", float(errors[metric]))
        assert errors[metric] < limit, errors
