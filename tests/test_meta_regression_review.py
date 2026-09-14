from __future__ import annotations

import itertools
import json
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma
from meta_analyze.estimators import estimate_meta_regression_tau2

REFERENCE_DIRECTORY = Path(__file__).parent / "reference"
REGRESSION_REFERENCE_FILES = (
    "meta_regression_metafor.json",
    "meta_regression_influence_metafor.json",
    "meta_regression_collinearity_metafor.json",
    "meta_regression_contrasts_metafor.json",
)


@pytest.mark.parametrize("filename", REGRESSION_REFERENCE_FILES)
def test_meta_regression_references_record_pinned_environment(filename: str) -> None:
    reference = json.loads((REFERENCE_DIRECTORY / filename).read_text(encoding="utf-8"))

    assert reference["r_version"] == "R version 4.6.1 (2026-06-24 ucrt)"
    assert reference["metafor_version"] == "5.0.1"
    assert reference["jsonlite_version"] == "2.0.0"


def _exact_extreme_dl_tau2(precision_exponent: int) -> float:
    small = Fraction(1, 10**precision_exponent)
    weights = (Fraction(1), Fraction(1), small, small)
    moderator = tuple(Fraction(value) for value in range(4))
    effects = tuple(Fraction(value) for value in (0, 1, 10, -10))

    gram_00 = sum(weights)
    gram_01 = sum(
        weight * value for weight, value in zip(weights, moderator, strict=True)
    )
    gram_11 = sum(
        weight * value * value for weight, value in zip(weights, moderator, strict=True)
    )
    determinant = gram_00 * gram_11 - gram_01 * gram_01
    inverse = (
        (gram_11 / determinant, -gram_01 / determinant),
        (-gram_01 / determinant, gram_00 / determinant),
    )
    rhs = (
        sum(weight * effect for weight, effect in zip(weights, effects, strict=True)),
        sum(
            weight * value * effect
            for weight, value, effect in zip(weights, moderator, effects, strict=True)
        ),
    )
    coefficients = (
        inverse[0][0] * rhs[0] + inverse[0][1] * rhs[1],
        inverse[1][0] * rhs[0] + inverse[1][1] * rhs[1],
    )
    q_scaled = sum(
        weight * (effect - coefficients[0] - coefficients[1] * value) ** 2
        for weight, value, effect in zip(weights, moderator, effects, strict=True)
    )

    squared_gram = (
        (
            sum(weight * weight for weight in weights),
            sum(
                weight * weight * value
                for weight, value in zip(weights, moderator, strict=True)
            ),
        ),
        (
            sum(
                weight * weight * value
                for weight, value in zip(weights, moderator, strict=True)
            ),
            sum(
                weight * weight * value * value
                for weight, value in zip(weights, moderator, strict=True)
            ),
        ),
    )
    trace_product = sum(
        inverse[row][column] * squared_gram[column][row]
        for row in range(2)
        for column in range(2)
    )
    projection_trace = sum(weights) - trace_product
    residual_df = 2
    return float((q_scaled - residual_df * small) / projection_trace)


@pytest.mark.parametrize("precision_exponent", [18, 100, 300])
def test_generalized_dl_extreme_precision_matches_exact_fraction(
    precision_exponent: int,
) -> None:
    small = 10.0**-precision_exponent
    effect = np.asarray([0.0, 1.0, 10.0, -10.0])
    variance = np.asarray([small, small, 1.0, 1.0])
    design = np.column_stack([np.ones(4), np.arange(4.0)])
    expected = _exact_extreme_dl_tau2(precision_exponent)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        estimate = estimate_meta_regression_tau2(
            effect,
            variance,
            design,
            method="DL",
            atol=1e-12,
            max_iter=1000,
        )
        result = ma.meta_regression(
            effect=effect,
            variance=variance,
            moderators={"x": design[:, 1]},
            model="mixed",
            tau2_method="DL",
            atol=1e-12,
        )

    assert estimate.value == pytest.approx(expected, rel=3e-14)
    assert result.tau2 == pytest.approx(expected, rel=3e-14)
    assert result.leave_one_out().table["refit_success"].all()


def test_generalized_dl_extreme_precision_is_row_order_invariant() -> None:
    effect = np.asarray([0.0, 1.0, 10.0, -10.0])
    variance = np.asarray([1e-300, 1e-300, 1.0, 1.0])
    design = np.column_stack([np.ones(4), np.arange(4.0)])
    expected = _exact_extreme_dl_tau2(300)

    estimates = []
    for permutation in itertools.permutations(range(4)):
        order = np.asarray(permutation)
        estimates.append(
            estimate_meta_regression_tau2(
                effect[order],
                variance[order],
                design[order],
                method="DL",
                atol=1e-12,
                max_iter=1000,
            ).value
        )

    np.testing.assert_allclose(estimates, expected, rtol=3e-14, atol=0.0)


def _review_regression_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moderator = np.linspace(-2.0, 2.0, 12)
    effect = np.asarray([0.2, 1.4, -0.3, 1.6, 0.1, 1.8, -0.2, 2.0, 0.3, 2.2, -0.1, 2.4])
    variance = np.asarray(
        [0.04, 0.06, 0.05, 0.08, 0.045, 0.07, 0.055, 0.09, 0.05, 0.075, 0.065, 0.06]
    )
    return effect, variance, moderator


@pytest.mark.parametrize(
    ("model", "tau2_method"),
    [
        ("common", None),
        ("mixed", "DL"),
        ("mixed", "PM"),
        ("mixed", "REML"),
    ],
)
@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_moderator_units_preserve_fit_prediction_and_diagnostics(
    model: str,
    tau2_method: str | None,
    scale: float,
) -> None:
    effect, variance, moderator = _review_regression_inputs()
    options: dict[str, object] = {"model": model, "atol": 1e-12}
    if tau2_method is not None:
        options["tau2_method"] = tau2_method
    original = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        **options,  # type: ignore[arg-type]
    )
    rescaled = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator * scale},
        **options,  # type: ignore[arg-type]
    )

    original_coefficients = original.coefficients["estimate"].to_numpy()
    rescaled_coefficients = rescaled.coefficients["estimate"].to_numpy()
    assert rescaled_coefficients[0] == pytest.approx(
        original_coefficients[0], rel=3e-12, abs=3e-14
    )
    assert rescaled_coefficients[1] * scale == pytest.approx(
        original_coefficients[1], rel=3e-12, abs=3e-14
    )
    original_covariance = original.coefficient_covariance.to_numpy()
    rescaled_covariance = rescaled.coefficient_covariance.to_numpy()
    assert rescaled_covariance[0, 0] == pytest.approx(
        original_covariance[0, 0], rel=5e-12
    )
    assert rescaled_covariance[0, 1] * scale == pytest.approx(
        original_covariance[0, 1], rel=5e-12, abs=5e-14
    )
    assert rescaled_covariance[1, 1] * scale * scale == pytest.approx(
        original_covariance[1, 1], rel=5e-12
    )
    assert rescaled.tau2 == pytest.approx(original.tau2, rel=3e-11, abs=3e-12)
    assert rescaled.heterogeneity.q == pytest.approx(
        original.heterogeneity.q, rel=5e-12
    )
    np.testing.assert_allclose(
        rescaled.study_results["residual"],
        original.study_results["residual"],
        rtol=3e-12,
        atol=3e-14,
    )

    prediction_values = np.asarray([-1.5, 0.0, 1.5])
    original_prediction = original.predict(pd.DataFrame({"x": prediction_values}))
    rescaled_prediction = rescaled.predict(
        pd.DataFrame({"x": prediction_values * scale})
    )
    np.testing.assert_allclose(
        rescaled_prediction,
        original_prediction,
        rtol=5e-12,
        atol=5e-14,
    )
    original_contrast = original.contrast({"x": 1.0})
    rescaled_contrast = rescaled.contrast({"x": scale})
    np.testing.assert_allclose(
        rescaled_contrast.table[["estimate", "standard_error", "statistic", "pvalue"]],
        original_contrast.table[["estimate", "standard_error", "statistic", "pvalue"]],
        rtol=5e-12,
        atol=5e-14,
    )
    original_collinearity = original.collinearity()
    rescaled_collinearity = rescaled.collinearity()
    np.testing.assert_allclose(
        rescaled_collinearity.condition_indices["condition_index"],
        original_collinearity.condition_indices["condition_index"],
        rtol=5e-12,
        atol=5e-14,
    )


@pytest.mark.parametrize("tau2_method", ["DL", "PM", "REML"])
def test_intercept_model_preserves_residual_geometry_after_large_effect_shift(
    tau2_method: str,
) -> None:
    moderator = np.linspace(-2.0, 2.0, 12)
    effect = np.asarray(
        [0.0, 4.0, -2.0, 6.0, 0.0, 8.0, -4.0, 10.0, 2.0, 12.0, -2.0, 14.0]
    )
    variance = np.asarray(
        [0.04, 0.06, 0.05, 0.08, 0.045, 0.07, 0.055, 0.09, 0.05, 0.075, 0.065, 0.06]
    )
    shift = 1e16
    shifted_effect = effect + shift
    assert np.array_equal(shifted_effect - shift, effect)

    original = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        model="mixed",
        tau2_method=tau2_method,
        atol=1e-12,
    )
    shifted = ma.meta_regression(
        effect=shifted_effect,
        variance=variance,
        moderators={"x": moderator},
        model="mixed",
        tau2_method=tau2_method,
        atol=1e-12,
    )

    assert shifted.coefficients.loc[1, "estimate"] == pytest.approx(
        original.coefficients.loc[1, "estimate"], rel=3e-12
    )
    assert shifted.tau2 == pytest.approx(original.tau2, rel=5e-10, abs=5e-10)
    assert shifted.heterogeneity.q == pytest.approx(original.heterogeneity.q, rel=3e-13)
    np.testing.assert_allclose(
        shifted.study_results["residual"],
        original.study_results["residual"],
        rtol=3e-12,
        atol=3e-13,
    )


def test_contrast_row_scaling_preserves_individual_and_joint_inference() -> None:
    effect, variance, moderator = _review_regression_inputs()
    result = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        model="common",
    )
    original = result.contrast(
        {
            "intercept": {"intercept": 1.0},
            "slope": {"x": 1.0},
        },
        rhs={"intercept": 0.1, "slope": 0.25},
    )
    row_scales = np.asarray([1e-200, 1e200])
    rescaled = result.contrast(
        {
            "intercept": {"intercept": row_scales[0]},
            "slope": {"x": row_scales[1]},
        },
        rhs={
            "intercept": 0.1 * row_scales[0],
            "slope": 0.25 * row_scales[1],
        },
    )

    np.testing.assert_allclose(
        rescaled.table["estimate"] / row_scales,
        original.table["estimate"],
        rtol=3e-13,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        rescaled.table["standard_error"] / row_scales,
        original.table["standard_error"],
        rtol=3e-13,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        rescaled.table["estimate_minus_rhs"] / row_scales,
        original.table["estimate_minus_rhs"],
        rtol=3e-13,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        rescaled.table[["statistic", "pvalue"]],
        original.table[["statistic", "pvalue"]],
        rtol=3e-13,
        atol=3e-15,
    )
    assert rescaled.joint_test.statistic == pytest.approx(
        original.joint_test.statistic, rel=5e-13
    )
    assert rescaled.joint_test.pvalue == pytest.approx(
        original.joint_test.pvalue, rel=5e-13
    )
