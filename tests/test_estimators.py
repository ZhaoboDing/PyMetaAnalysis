from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import meta_analyze as ma
import meta_analyze.estimators.meta_regression as meta_regression_module
import meta_analyze.estimators.tau2 as tau2_module
from meta_analyze.estimators import (
    estimate_meta_regression_tau2,
    estimate_tau2,
    fit_mantel_haenszel,
)

EFFECT = np.array([-0.4, 0.1, 0.5, 1.2, 1.8], dtype=float)
VARIANCE = np.array([0.04, 0.09, 0.05, 0.16, 0.08], dtype=float)


def test_dl_matches_closed_form_calculation() -> None:
    common_weights = 1.0 / VARIANCE
    common_estimate = np.sum(common_weights * EFFECT) / np.sum(common_weights)
    q = np.sum(common_weights * (EFFECT - common_estimate) ** 2)
    c = np.sum(common_weights) - np.sum(common_weights**2) / np.sum(common_weights)
    expected_tau2 = max(0.0, (q - (len(EFFECT) - 1)) / c)

    result = ma.meta_analysis(
        effect=EFFECT,
        variance=VARIANCE,
        model="random",
        tau2_method="DL",
    )

    assert result.tau2 == pytest.approx(expected_tau2, rel=1e-12, abs=1e-12)
    assert result.diagnostics.converged
    assert result.study_results["normalized_weight"].sum() == pytest.approx(1.0)


def test_pm_solution_satisfies_generalized_q_equation() -> None:
    result = ma.meta_analysis(
        effect=EFFECT,
        variance=VARIANCE,
        model="random",
        tau2_method="PM",
    )
    weights = 1.0 / (VARIANCE + result.tau2)
    estimate = np.sum(weights * EFFECT) / np.sum(weights)
    generalized_q = np.sum(weights * (EFFECT - estimate) ** 2)

    assert result.tau2 > 0.0
    assert generalized_q == pytest.approx(len(EFFECT) - 1, rel=1e-9, abs=1e-9)


def test_reml_solution_satisfies_restricted_likelihood_score() -> None:
    result = ma.meta_analysis(
        effect=EFFECT,
        variance=VARIANCE,
        model="random",
        tau2_method="REML",
    )
    weights = 1.0 / (VARIANCE + result.tau2)
    estimate = np.sum(weights * EFFECT) / np.sum(weights)
    residual = EFFECT - estimate
    score = 0.5 * (
        np.sum(weights**2 * residual**2)
        - (np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    )

    assert result.tau2 > 0.0
    assert score == pytest.approx(0.0, abs=1e-8)
    assert result.method.tau2_method == "REML"


def test_tau_estimators_return_zero_at_boundary() -> None:
    for method in ("DL", "PM", "REML"):
        result = ma.meta_analysis(
            effect=[0.2, 0.2, 0.2],
            variance=[0.1, 0.2, 0.3],
            model="random",
            tau2_method=method,
        )
        assert result.tau2 == pytest.approx(0.0)
        assert result.diagnostics.tau2_at_boundary


@pytest.mark.parametrize("method", ["DL", "PM", "REML"])
def test_tau2_estimator_rejects_a_single_study(method: str) -> None:
    with pytest.raises(ma.InsufficientStudiesError, match="at least two"):
        estimate_tau2(
            np.asarray([0.2]),
            np.asarray([0.1]),
            method=method,
        )


@pytest.mark.parametrize("method", ["DL", "PM", "REML"])
def test_meta_regression_tau2_requires_positive_residual_df(method: str) -> None:
    with pytest.raises(
        ma.InsufficientStudiesError,
        match="positive residual degrees of freedom",
    ):
        estimate_meta_regression_tau2(
            np.asarray([0.1, 0.2]),
            np.asarray([0.04, 0.05]),
            np.eye(2),
            method=method,
            atol=1e-10,
            max_iter=1000,
        )


def _positive_small_root(
    *_: object,
    **__: object,
) -> tuple[float, SimpleNamespace]:
    return 5e-11, SimpleNamespace(converged=True, iterations=3)


def test_positive_tau2_root_below_atol_is_not_a_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tau2_module, "brentq", _positive_small_root)

    estimate = estimate_tau2(
        EFFECT,
        VARIANCE,
        method="PM",
        atol=1e-10,
    )

    assert estimate.value == pytest.approx(5e-11)
    assert estimate.boundary is False


def test_positive_meta_regression_tau2_root_below_atol_is_not_a_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(meta_regression_module, "brentq", _positive_small_root)
    design = np.column_stack([np.ones(5), np.arange(5.0)])

    estimate = estimate_meta_regression_tau2(
        np.asarray([0.0, 3.0, -1.0, 4.0, 0.0]),
        VARIANCE,
        design,
        method="PM",
        atol=1e-10,
        max_iter=1000,
    )

    assert estimate.value == pytest.approx(5e-11)
    assert estimate.boundary is False


def test_mantel_haenszel_rejects_empty_and_all_zero_strata() -> None:
    empty = np.asarray([], dtype=np.float64)
    with pytest.raises(ma.InsufficientStudiesError, match="at least one"):
        fit_mantel_haenszel(
            empty,
            empty,
            empty,
            empty,
            measure="RR",
            confidence_level=0.95,
        )

    with pytest.raises(ma.InvalidStudyDataError, match="zero total.*row positions.*1"):
        fit_mantel_haenszel(
            np.asarray([1.0, 0.0]),
            np.asarray([2.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([2.0, 0.0]),
            measure="RR",
            confidence_level=0.95,
        )


def test_hartung_knapp_adhoc_never_has_smaller_se_than_classic() -> None:
    effect = [0.10, 0.11, 0.12, 0.13]
    variance = [0.04, 0.04, 0.04, 0.04]
    classic = ma.meta_analysis(
        effect=effect,
        variance=variance,
        model="random",
        ci_method="normal",
    )
    hk = ma.meta_analysis(
        effect=effect,
        variance=variance,
        model="random",
        ci_method="hartung_knapp",
    )
    protected = ma.meta_analysis(
        effect=effect,
        variance=variance,
        model="random",
        ci_method="hartung_knapp_adhoc",
    )

    assert hk.standard_error < classic.standard_error
    assert protected.standard_error == pytest.approx(classic.standard_error)
    assert any("below the classic variance" in warning for warning in hk.warnings)


def test_hartung_knapp_zero_variance_is_reported() -> None:
    result = ma.meta_analysis(
        effect=[0.2, 0.2, 0.2],
        variance=[0.1, 0.2, 0.3],
        model="random",
        ci_method="hartung_knapp",
    )

    assert result.standard_error == 0.0
    assert result.ci == pytest.approx((0.2, 0.2))
    assert any("variance is zero" in warning for warning in result.warnings)


def test_random_effects_prediction_interval_requires_three_studies() -> None:
    two = ma.meta_analysis(effect=[0.1, 0.4], variance=[0.01, 0.02], model="random")
    two_hk = ma.meta_analysis(
        effect=[0.1, 0.4],
        variance=[0.01, 0.02],
        model="random",
        ci_method="hartung_knapp",
    )
    three = ma.meta_analysis(
        effect=[0.1, 0.4, 0.8], variance=[0.01, 0.02, 0.03], model="random"
    )

    assert two.prediction_interval is None
    assert two.method.prediction_interval_method is None
    assert two_hk.prediction_interval is None
    assert two_hk.method.prediction_interval_method is None
    assert two_hk.method.ci_method == "hartung_knapp"
    assert any("at least three" in warning for warning in two.warnings)
    assert three.prediction_interval is not None
    assert three.method.prediction_interval_method == "HTS"
    assert "Prediction interval:" in str(three.summary())
    assert "tau^2:" in str(three.summary())
    assert any("fewer than five" in warning for warning in three.warnings)

    five = ma.meta_analysis(
        effect=[0.1, 0.3, 0.5, 0.7, 0.9],
        variance=[0.01, 0.02, 0.03, 0.04, 0.05],
        model="random",
    )
    assert five.prediction_interval is not None
    assert not any("fewer than five" in warning for warning in five.warnings)


def test_hartung_knapp_is_rejected_for_common_effect_model() -> None:
    with pytest.raises(ma.UnsupportedMethodError, match="only supported"):
        ma.meta_analysis(
            effect=[0.1, 0.2],
            variance=[0.01, 0.02],
            model="common",
            ci_method="hartung_knapp",
        )
