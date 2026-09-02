from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_DATA = pd.read_csv(REFERENCE_DIR / "small_study_effects_input.csv")
REFERENCE = json.loads(
    (REFERENCE_DIR / "small_study_effects_metafor.json").read_text(encoding="utf-8")
)


def _reference_result() -> ma.MetaAnalysisResult:
    return ma.meta_analysis(
        REFERENCE_DATA,
        effect="effect",
        variance="variance",
        study="study",
        model="common",
    )


def test_egger_matches_direct_weighted_regression_calculation() -> None:
    effect = np.asarray([-0.4, 0.1, 0.5, 1.2, 1.8])
    variance = np.asarray([0.04, 0.09, 0.05, 0.16, 0.08])
    standard_error = np.sqrt(variance)
    design = np.column_stack((np.ones(len(effect)), standard_error))
    weights = np.diag(1.0 / variance)
    information = design.T @ weights @ design
    coefficients = np.linalg.solve(information, design.T @ weights @ effect)
    residual = effect - design @ coefficients
    dispersion = float(residual @ weights @ residual / (len(effect) - 2))
    covariance = dispersion * np.linalg.inv(information)
    coefficient_se = np.sqrt(np.diag(covariance))
    expected_statistic = coefficients[1] / coefficient_se[1]
    expected_pvalue = 2.0 * t.sf(abs(expected_statistic), df=len(effect) - 2)

    result = ma.meta_analysis(
        effect=effect,
        variance=variance,
        model="common",
    ).egger_test()

    assert result.limit_estimate == pytest.approx(coefficients[0], rel=2e-14)
    assert result.intercept == pytest.approx(coefficients[1], rel=2e-14)
    assert result.limit_standard_error == pytest.approx(coefficient_se[0], rel=2e-14)
    assert result.intercept_standard_error == pytest.approx(
        coefficient_se[1], rel=2e-14
    )
    assert result.statistic == pytest.approx(expected_statistic, rel=2e-14)
    assert result.pvalue == pytest.approx(expected_pvalue, rel=2e-14)
    assert result.df == 3


def test_egger_matches_metafor_reference() -> None:
    result = _reference_result().egger_test()

    assert isinstance(result, ma.EggerTestResult)
    assert result.k == REFERENCE["k"]
    assert result.confidence_level == pytest.approx(REFERENCE["confidence_level"])
    assert result.intercept == pytest.approx(REFERENCE["intercept"], rel=5e-13)
    assert result.intercept_standard_error == pytest.approx(
        REFERENCE["intercept_standard_error"], rel=5e-13
    )
    assert result.intercept_ci == pytest.approx(REFERENCE["intercept_ci"], rel=5e-13)
    assert result.statistic == pytest.approx(REFERENCE["statistic"], rel=5e-13)
    assert result.statistic_name == "t"
    assert result.distribution == "t"
    assert result.df == REFERENCE["df"]
    assert result.pvalue == pytest.approx(REFERENCE["pvalue"], rel=5e-13)
    assert result.limit_estimate == pytest.approx(
        REFERENCE["limit_estimate"], rel=5e-13
    )
    assert result.limit_standard_error == pytest.approx(
        REFERENCE["limit_standard_error"], rel=5e-13
    )
    assert result.limit_ci == pytest.approx(REFERENCE["limit_ci"], rel=5e-13)
    assert result.method == "egger"
    assert result.model == "weighted_regression_multiplicative_dispersion"
    assert result.predictor == "standard_error"
    assert np.isfinite(result.condition_number)


def test_egger_is_invariant_to_row_order_and_effect_location() -> None:
    original = _reference_result().egger_test()
    reordered_data = REFERENCE_DATA.iloc[::-1].reset_index(drop=True)
    reordered = ma.meta_analysis(
        reordered_data,
        effect="effect",
        variance="variance",
        model="common",
    ).egger_test()
    shifted = ma.meta_analysis(
        effect=REFERENCE_DATA["effect"] + 2.5,
        variance=REFERENCE_DATA["variance"],
        model="common",
    ).egger_test()

    assert reordered.intercept == pytest.approx(original.intercept, rel=2e-14)
    assert reordered.statistic == pytest.approx(original.statistic, rel=2e-14)
    assert reordered.limit_estimate == pytest.approx(original.limit_estimate, rel=2e-14)
    assert shifted.intercept == pytest.approx(original.intercept, abs=2e-14)
    assert shifted.intercept_standard_error == pytest.approx(
        original.intercept_standard_error, rel=2e-14
    )
    assert shifted.statistic == pytest.approx(original.statistic, rel=2e-14)
    assert shifted.limit_estimate == pytest.approx(
        original.limit_estimate + 2.5, rel=2e-14
    )


def test_egger_rescales_the_snd_intercept_with_standard_errors() -> None:
    original = _reference_result().egger_test()
    scaled = ma.meta_analysis(
        effect=REFERENCE_DATA["effect"],
        variance=REFERENCE_DATA["variance"] * 100.0,
        model="common",
    ).egger_test()

    assert scaled.intercept == pytest.approx(original.intercept / 10.0, rel=2e-14)
    assert scaled.intercept_standard_error == pytest.approx(
        original.intercept_standard_error / 10.0, rel=2e-14
    )
    assert scaled.statistic == pytest.approx(original.statistic, rel=2e-14)
    assert scaled.pvalue == pytest.approx(original.pvalue, rel=2e-14)
    assert scaled.limit_estimate == pytest.approx(original.limit_estimate, rel=2e-14)
    assert scaled.limit_standard_error == pytest.approx(
        original.limit_standard_error, rel=2e-14
    )


def test_egger_uses_only_included_rows() -> None:
    data = pd.concat(
        [
            REFERENCE_DATA,
            pd.DataFrame([{"study": "missing", "effect": np.nan, "variance": 0.2}]),
        ],
        ignore_index=True,
    )

    result = ma.meta_analysis(
        data,
        effect="effect",
        variance="variance",
        study="study",
        model="common",
        missing="drop",
    ).egger_test()
    expected = _reference_result().egger_test()

    assert result.k == len(REFERENCE_DATA)
    assert result.intercept == pytest.approx(expected.intercept, rel=2e-14)
    assert result.limit_estimate == pytest.approx(expected.limit_estimate, rel=2e-14)


def test_egger_reuses_or_overrides_fitted_confidence_level() -> None:
    fitted = ma.meta_analysis(
        REFERENCE_DATA,
        effect="effect",
        variance="variance",
        model="common",
        confidence_level=0.9,
    )

    inherited = fitted.egger_test()
    overridden = fitted.egger_test(confidence_level=0.8)

    assert inherited.confidence_level == 0.9
    assert overridden.confidence_level == 0.8
    assert overridden.intercept_ci_low > inherited.intercept_ci_low
    assert overridden.intercept_ci_high < inherited.intercept_ci_high
    assert overridden.limit_ci_low > inherited.limit_ci_low
    assert overridden.limit_ci_high < inherited.limit_ci_high


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, np.nan, np.inf, True, "0.95"])
def test_egger_rejects_invalid_confidence_level(confidence_level: object) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="between 0 and 1"):
        _reference_result().egger_test(confidence_level=confidence_level)  # type: ignore[arg-type]


def test_egger_requires_three_studies_and_varying_standard_errors() -> None:
    two_studies = ma.meta_analysis(
        effect=[0.1, 0.2],
        variance=[0.1, 0.2],
        model="common",
    )
    equal_standard_errors = ma.meta_analysis(
        effect=[0.1, 0.2, 0.4, 0.5],
        variance=[0.1, 0.1, 0.1, 0.1],
        model="common",
    )

    with pytest.raises(ma.InsufficientStudiesError, match="at least three"):
        two_studies.egger_test()
    with pytest.raises(ma.InvalidStudyDataError, match="standard errors"):
        equal_standard_errors.egger_test()


def test_egger_records_interpretation_and_measure_specific_warnings() -> None:
    generic = _reference_result().egger_test()
    odds_ratio = replace(
        _reference_result(),
        measure="OR",
        effect_scale="log",
        display_scale="exp",
    ).egger_test()
    standardized_difference = replace(
        _reference_result(),
        measure="SMD",
    ).egger_test()
    small = ma.meta_analysis(
        effect=[-0.4, 0.1, 0.5, 1.2, 1.8],
        variance=[0.04, 0.09, 0.05, 0.16, 0.08],
        model="common",
    ).egger_test()

    assert len(generic.warnings) == 1
    assert (
        "does not by itself establish or exclude publication bias"
        in generic.warnings[0]
    )
    assert any("odds ratios" in warning for warning in odds_ratio.warnings)
    assert any(
        "standardized mean differences" in warning
        for warning in standardized_difference.warnings
    )
    assert any("fewer than ten" in warning for warning in small.warnings)
    assert odds_ratio.display_limit_estimate == pytest.approx(
        math.exp(odds_ratio.limit_estimate)
    )
    assert odds_ratio.display_limit_ci == pytest.approx(
        tuple(math.exp(value) for value in odds_ratio.limit_ci)
    )


def test_egger_result_is_immutable_printable_and_machine_readable() -> None:
    result = _reference_result().egger_test()
    payload = result.to_dict()

    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0.5  # type: ignore[misc]
    assert payload["statistic_name"] == "t"
    assert payload["distribution"] == "t"
    assert payload["display_limit_ci"] == result.display_limit_ci
    assert payload["warnings"] == result.warnings
    assert "Egger regression test for funnel-plot asymmetry" in str(result)
    assert "weighted regression with multiplicative dispersion" in str(result)
