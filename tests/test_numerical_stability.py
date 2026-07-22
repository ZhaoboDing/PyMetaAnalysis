from __future__ import annotations

import numpy as np
import pytest

import meta_analyze as ma


def test_common_model_scales_weights_at_float64_minimum_variance() -> None:
    tiny = np.finfo(np.float64).tiny
    variances = tiny * np.array([1.0, 2.0, 4.0])

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = ma.meta_analysis(
            effect=[1.0, 1.0, 1.0],
            variance=variances,
            model="common",
        )

    expected_weights = np.array([4.0, 2.0, 1.0]) / 7.0
    expected_variance = tiny / (1.0 + 0.5 + 0.25)
    np.testing.assert_allclose(
        result.study_results["normalized_weight"],
        expected_weights,
        rtol=1e-15,
        atol=0.0,
    )
    assert result.estimate == 1.0
    assert result.standard_error == pytest.approx(np.sqrt(expected_variance), rel=1e-15)
    assert result.q == 0.0


def test_common_model_handles_extreme_weight_imbalance() -> None:
    result = ma.meta_analysis(
        effect=[0.25, 10.0, -10.0],
        variance=[1e-300, 1e-150, 1.0],
        model="common",
    )
    weights = result.study_results["normalized_weight"].to_numpy()

    assert result.estimate == pytest.approx(0.25, abs=1e-14)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)
    assert weights[0] == 1.0
    assert weights[1] == pytest.approx(1e-150, rel=1e-15)
    assert np.isfinite(result.q)


def test_common_model_with_very_large_variances_matches_scaled_oracle() -> None:
    result = ma.meta_analysis(
        effect=[1.0, 2.0, 3.0],
        variance=[1e300, 2e300, 5e300],
        model="common",
    )
    expected_weights = np.array([1.0, 0.5, 0.2]) / 1.7

    np.testing.assert_allclose(
        result.study_results["normalized_weight"],
        expected_weights,
        rtol=1e-15,
        atol=0.0,
    )
    assert result.estimate == pytest.approx(26.0 / 17.0, rel=1e-15)
    assert np.isfinite(result.standard_error)
    assert np.isfinite(result.ci_low)
    assert np.isfinite(result.ci_high)


@pytest.mark.parametrize("tau2_method", ["DL", "PM", "REML"])
def test_random_models_remain_translation_equivariant_at_large_location(
    tau2_method: str,
) -> None:
    effect = np.array([-0.4, 0.1, 0.5, 1.2, 1.8])
    variance = np.array([0.04, 0.09, 0.05, 0.16, 0.08])
    offset = 1e9
    original = ma.meta_analysis(
        effect=effect,
        variance=variance,
        model="random",
        tau2_method=tau2_method,
    )
    shifted = ma.meta_analysis(
        effect=effect + offset,
        variance=variance,
        model="random",
        tau2_method=tau2_method,
    )

    assert shifted.estimate - offset == pytest.approx(original.estimate, abs=5e-7)
    assert shifted.standard_error == pytest.approx(original.standard_error, rel=2e-7)
    assert shifted.tau2 == pytest.approx(original.tau2, rel=3e-7)
    assert shifted.q == pytest.approx(original.q, rel=3e-7)


@pytest.mark.parametrize(
    ("measure", "method"),
    [("OR", "IV"), ("RR", "IV"), ("RD", "IV"), ("OR", "MH"), ("RR", "MH")],
)
def test_large_binary_counts_preserve_scale_invariant_estimates(
    measure: str,
    method: str,
) -> None:
    inputs = {
        "event_treat": np.array([12, 5, 20, 7]),
        "n_treat": np.array([100, 80, 120, 90]),
        "event_control": np.array([18, 9, 15, 10]),
        "n_control": np.array([110, 75, 130, 95]),
    }
    base = ma.meta_binary(**inputs, measure=measure, method=method, model="common")
    scaled = ma.meta_binary(
        **{name: values * 1_000_000 for name, values in inputs.items()},
        measure=measure,
        method=method,
        model="common",
    )

    assert scaled.estimate == pytest.approx(base.estimate, abs=2e-15)
    assert np.isfinite(scaled.standard_error)
    assert np.isfinite(scaled.q)
    assert scaled.study_results["normalized_weight"].sum() == pytest.approx(
        1.0, abs=2e-15
    )


@pytest.mark.parametrize("measure", ["MD", "SMD"])
def test_continuous_effects_are_stable_after_a_large_location_shift(
    measure: str,
) -> None:
    inputs = {
        "mean_treat": np.array([2.1, 2.4, 4.0]),
        "sd_treat": np.array([1.0, 1.1, 1.2]),
        "n_treat": np.array([1_000_000, 1_100_000, 1_200_000]),
        "mean_control": np.array([2.0, 2.0, 3.0]),
        "sd_control": np.array([1.0, 1.0, 1.1]),
        "n_control": np.array([900_000, 1_000_000, 1_100_000]),
    }
    original = ma.meta_continuous(**inputs, measure=measure, model="common")
    shifted_inputs = dict(inputs)
    shifted_inputs["mean_treat"] = inputs["mean_treat"] + 1e9
    shifted_inputs["mean_control"] = inputs["mean_control"] + 1e9
    shifted = ma.meta_continuous(
        **shifted_inputs,
        measure=measure,
        model="common",
    )

    np.testing.assert_allclose(
        shifted.study_results["effect"],
        original.study_results["effect"],
        rtol=0.0,
        atol=1e-7,
    )
    assert shifted.estimate == pytest.approx(original.estimate, abs=1e-7)
    assert np.isfinite(shifted.standard_error)


@pytest.mark.parametrize("tau2_method", ["DL", "PM", "REML"])
def test_mixed_meta_regression_is_stable_after_large_moderator_rescaling(
    tau2_method: str,
) -> None:
    moderator = np.linspace(-2.0, 2.0, 12)
    effect = np.array([0.2, 1.4, -0.3, 1.6, 0.1, 1.8, -0.2, 2.0, 0.3, 2.2, -0.1, 2.4])
    variance = np.array(
        [0.04, 0.06, 0.05, 0.08, 0.045, 0.07, 0.055, 0.09, 0.05, 0.075, 0.065, 0.06]
    )
    scale = 1e9
    original = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        model="mixed",
        tau2_method=tau2_method,
    )
    rescaled = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator * scale},
        model="mixed",
        tau2_method=tau2_method,
    )

    assert rescaled.coefficients.loc[0, "estimate"] == pytest.approx(
        original.coefficients.loc[0, "estimate"], rel=1e-12, abs=1e-12
    )
    assert rescaled.coefficients.loc[1, "estimate"] * scale == pytest.approx(
        original.coefficients.loc[1, "estimate"], rel=1e-12, abs=1e-12
    )
    assert rescaled.tau2 == pytest.approx(original.tau2, rel=1e-12, abs=1e-12)
    assert rescaled.heterogeneity.q == pytest.approx(
        original.heterogeneity.q, rel=1e-12, abs=1e-12
    )
    np.testing.assert_allclose(
        rescaled.study_results["fitted_value"],
        original.study_results["fitted_value"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert any("high condition number" in warning for warning in rescaled.warnings)


@pytest.mark.parametrize("tau2_method", ["DL", "PM", "REML"])
def test_mixed_meta_regression_is_stable_after_large_effect_shift(
    tau2_method: str,
) -> None:
    moderator = np.linspace(-2.0, 2.0, 12)
    effect = np.array([0.2, 1.4, -0.3, 1.6, 0.1, 1.8, -0.2, 2.0, 0.3, 2.2, -0.1, 2.4])
    variance = np.array(
        [0.04, 0.06, 0.05, 0.08, 0.045, 0.07, 0.055, 0.09, 0.05, 0.075, 0.065, 0.06]
    )
    shift = 1e9
    original = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        model="mixed",
        tau2_method=tau2_method,
    )
    shifted = ma.meta_regression(
        effect=effect + shift,
        variance=variance,
        moderators={"x": moderator},
        model="mixed",
        tau2_method=tau2_method,
    )

    assert shifted.coefficients.loc[0, "estimate"] - shift == pytest.approx(
        original.coefficients.loc[0, "estimate"], abs=5e-7
    )
    assert shifted.coefficients.loc[1, "estimate"] == pytest.approx(
        original.coefficients.loc[1, "estimate"], rel=5e-7
    )
    assert shifted.tau2 == pytest.approx(original.tau2, rel=5e-8)
    assert shifted.heterogeneity.q == pytest.approx(original.heterogeneity.q, rel=3e-8)
    np.testing.assert_allclose(
        shifted.study_results["fitted_value"] - shift,
        original.study_results["fitted_value"],
        rtol=0.0,
        atol=5e-7,
    )
