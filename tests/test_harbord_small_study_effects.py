from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_DATA = pd.read_csv(REFERENCE_DIR / "peters_small_study_effects_input.csv")
REFERENCE = json.loads(
    (REFERENCE_DIR / "harbord_small_study_effects_meta.json").read_text(
        encoding="utf-8"
    )
)
# R's qt() and older SciPy t.ppf() can differ slightly in the critical value.
# The interval-only tolerance covers that implementation difference while the
# coefficients, standard errors, statistic, and p-value remain tightly checked.
R_T_CI_ATOL = 5e-11


def _binary_result(
    data: pd.DataFrame = REFERENCE_DATA,
    *,
    method: str = "IV",
    model: str = "common",
    measure: str = "OR",
    **kwargs: object,
) -> ma.MetaAnalysisResult:
    return ma.meta_binary(
        data,
        event_treat="event_treat",
        n_treat="n_treat",
        event_control="event_control",
        n_control="n_control",
        study="study",
        method=method,
        model=model,
        measure=measure,
        **kwargs,  # type: ignore[arg-type]
    )


def test_harbord_matches_direct_standardized_score_regression() -> None:
    source = _binary_result()
    studies = source.study_results.loc[source.study_results["included"]]
    a = studies["event_treat"].to_numpy(dtype=float)
    c = studies["event_control"].to_numpy(dtype=float)
    n_treat = studies["n_treat"].to_numpy(dtype=float)
    n_control = studies["n_control"].to_numpy(dtype=float)
    total = n_treat + n_control
    events = a + c
    nonevents = total - events
    score = a - events * n_treat / total
    score_variance = (
        n_treat * n_control * events * nonevents / (total * total * (total - 1.0))
    )
    root_variance = np.sqrt(score_variance)
    response = score / root_variance
    design = np.column_stack((np.ones(len(studies)), root_variance))
    information = design.T @ design
    coefficients = np.linalg.solve(information, design.T @ response)
    residual = response - design @ coefficients
    dispersion = float(np.dot(residual, residual) / (len(studies) - 2))
    covariance = dispersion * np.linalg.inv(information)
    coefficient_se = np.sqrt(np.diag(covariance))

    result = source.harbord_test()

    assert result.intercept == pytest.approx(coefficients[0], rel=2e-14)
    assert result.limit_estimate == pytest.approx(coefficients[1], rel=2e-14)
    assert result.intercept_standard_error == pytest.approx(
        coefficient_se[0], rel=2e-14
    )
    assert result.limit_standard_error == pytest.approx(coefficient_se[1], rel=2e-14)
    assert result.statistic == pytest.approx(
        coefficients[0] / coefficient_se[0], rel=2e-14
    )
    assert result.residual_dispersion == pytest.approx(dispersion, rel=2e-14)


def test_harbord_matches_meta_reference() -> None:
    result = _binary_result().harbord_test()

    assert isinstance(result, ma.HarbordTestResult)
    assert REFERENCE["generated_by"] == "R meta"
    assert REFERENCE["meta_version"] == "8.5.0"
    assert result.k == REFERENCE["k"]
    assert result.confidence_level == pytest.approx(REFERENCE["confidence_level"])
    assert result.intercept == pytest.approx(REFERENCE["intercept"], rel=5e-13)
    assert result.intercept_standard_error == pytest.approx(
        REFERENCE["intercept_standard_error"], rel=5e-13
    )
    assert result.intercept_ci == pytest.approx(
        REFERENCE["intercept_ci"], rel=5e-13, abs=R_T_CI_ATOL
    )
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
    assert result.limit_ci == pytest.approx(
        REFERENCE["limit_ci"], rel=5e-13, abs=R_T_CI_ATOL
    )
    assert result.residual_dispersion == pytest.approx(
        REFERENCE["residual_dispersion"], rel=5e-13
    )
    assert result.method == "harbord"
    assert result.model == "efficient_score_regression_multiplicative_dispersion"
    assert result.response == "standardized_efficient_score"
    assert result.predictor == "sqrt_efficient_score_variance"
    assert result.weight_method == "efficient_score_variance_equivalent"
    assert result.uses_continuity_correction is False
    assert np.isfinite(result.condition_number)


@pytest.mark.parametrize(
    ("method", "model"),
    [("MH", "common"), ("IV", "common"), ("IV", "random"), ("Peto", "common")],
)
def test_harbord_is_independent_of_source_pooling_method(
    method: str, model: str
) -> None:
    expected = _binary_result().harbord_test()
    result = _binary_result(method=method, model=model).harbord_test()

    assert result.intercept == pytest.approx(expected.intercept, rel=2e-14)
    assert result.intercept_standard_error == pytest.approx(
        expected.intercept_standard_error, rel=2e-14
    )
    assert result.limit_estimate == pytest.approx(expected.limit_estimate, rel=2e-14)
    assert result.pvalue == pytest.approx(expected.pvalue, rel=2e-14)
    if method == "Peto":
        assert any("Peto pooling" in warning for warning in result.warnings)


def test_harbord_is_invariant_to_row_order_and_continuity_correction() -> None:
    expected = _binary_result().harbord_test()
    reordered = _binary_result(
        REFERENCE_DATA.iloc[::-1].reset_index(drop=True)
    ).harbord_test()
    corrected_everywhere = _binary_result(
        continuity_correction=1.0,
        correction_scope="all_studies",
    ).harbord_test()
    uncorrected = _binary_result(
        method="Peto",
        continuity_correction=0.0,
        correction_scope="none",
    ).harbord_test()

    for result in (reordered, corrected_everywhere, uncorrected):
        assert result.intercept == pytest.approx(expected.intercept, rel=2e-14)
        assert result.limit_estimate == pytest.approx(
            expected.limit_estimate, rel=2e-14
        )
        assert result.pvalue == pytest.approx(expected.pvalue, rel=2e-14)
        assert result.uses_continuity_correction is False


def test_harbord_ignores_excluded_rows() -> None:
    data = pd.concat(
        [
            REFERENCE_DATA,
            pd.DataFrame(
                [
                    {
                        "study": "double-zero",
                        "event_treat": 0,
                        "n_treat": 40,
                        "event_control": 0,
                        "n_control": 45,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = _binary_result(data).harbord_test()
    expected = _binary_result().harbord_test()

    assert result.k == len(REFERENCE_DATA)
    assert result.intercept == pytest.approx(expected.intercept, rel=2e-14)
    assert result.limit_estimate == pytest.approx(expected.limit_estimate, rel=2e-14)


def test_harbord_changes_sign_when_treatment_and_control_are_swapped() -> None:
    swapped = REFERENCE_DATA.rename(
        columns={
            "event_treat": "event_control",
            "n_treat": "n_control",
            "event_control": "event_treat",
            "n_control": "n_treat",
        }
    )
    original = _binary_result().harbord_test()
    reversed_result = _binary_result(swapped).harbord_test()

    assert reversed_result.intercept == pytest.approx(-original.intercept, rel=2e-14)
    assert reversed_result.statistic == pytest.approx(-original.statistic, rel=2e-14)
    assert reversed_result.pvalue == pytest.approx(original.pvalue, rel=2e-14)
    assert reversed_result.limit_estimate == pytest.approx(
        -original.limit_estimate, rel=2e-14
    )
    assert reversed_result.display_limit_estimate == pytest.approx(
        1.0 / original.display_limit_estimate, rel=2e-14
    )


def test_harbord_reuses_or_overrides_fitted_confidence_level() -> None:
    fitted = _binary_result(confidence_level=0.9)

    inherited = fitted.harbord_test()
    overridden = fitted.harbord_test(confidence_level=0.8)

    assert inherited.confidence_level == 0.9
    assert overridden.confidence_level == 0.8
    assert overridden.intercept_ci_low > inherited.intercept_ci_low
    assert overridden.intercept_ci_high < inherited.intercept_ci_high
    assert overridden.limit_ci_low > inherited.limit_ci_low
    assert overridden.limit_ci_high < inherited.limit_ci_high


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, np.nan, np.inf, True, "0.95"])
def test_harbord_rejects_invalid_confidence_level(confidence_level: object) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="between 0 and 1"):
        _binary_result().harbord_test(  # type: ignore[arg-type]
            confidence_level=confidence_level
        )


def test_harbord_requires_binary_odds_ratios_with_retained_counts() -> None:
    risk_ratio = _binary_result(measure="RR")
    generic = replace(
        ma.meta_analysis(
            effect=[0.1, 0.2, 0.3], variance=[0.01, 0.02, 0.03], model="common"
        ),
        measure="OR",
        effect_scale="log",
        display_scale="exp",
    )

    with pytest.raises(ma.UnsupportedMethodError, match="measure='OR'"):
        risk_ratio.harbord_test()
    with pytest.raises(ma.UnsupportedMethodError, match=r"meta_binary\(\)"):
        generic.harbord_test()


def test_harbord_requires_three_studies_and_varying_score_variances() -> None:
    two_studies = _binary_result(REFERENCE_DATA.iloc[:2].copy())
    equal_variances = pd.DataFrame(
        {
            "study": ["A", "B", "C", "D"],
            "event_treat": [4, 6, 8, 10],
            "n_treat": [50, 50, 50, 50],
            "event_control": [16, 14, 12, 10],
            "n_control": [50, 50, 50, 50],
        }
    )

    with pytest.raises(ma.InsufficientStudiesError, match="at least three"):
        two_studies.harbord_test()
    with pytest.raises(ma.InvalidStudyDataError, match="score variances"):
        _binary_result(equal_variances).harbord_test()


def test_harbord_records_few_study_and_interpretation_warnings() -> None:
    result = _binary_result(REFERENCE_DATA.iloc[:5].copy()).harbord_test()

    assert any("fewer than ten" in warning for warning in result.warnings)
    assert any("diagnostic-accuracy" in warning for warning in result.warnings)
    assert any(
        "does not by itself establish or exclude publication bias" in warning
        for warning in result.warnings
    )


def test_harbord_avoids_intermediate_overflow_for_large_counts() -> None:
    scale = 1e150
    data = pd.DataFrame(
        {
            "study": ["A", "B", "C", "D"],
            "event_treat": np.array([1.0, 1.2, 0.8, 1.1]) * scale,
            "n_treat": np.array([5.0, 6.0, 4.0, 7.0]) * scale,
            "event_control": np.array([1.5, 1.7, 1.4, 1.8]) * scale,
            "n_control": np.array([5.0, 5.0, 6.0, 8.0]) * scale,
        }
    )

    result = _binary_result(data).harbord_test()

    assert np.isfinite(result.intercept)
    assert np.isfinite(result.intercept_standard_error)
    assert np.isfinite(result.limit_estimate)
    assert np.isfinite(result.limit_standard_error)
    assert np.isfinite(result.residual_dispersion)
    assert np.isfinite(result.pvalue)


def test_harbord_result_is_immutable_printable_and_machine_readable() -> None:
    result = _binary_result().harbord_test()
    payload = result.to_dict()

    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0.5  # type: ignore[misc]
    assert payload["statistic_name"] == "t"
    assert payload["distribution"] == "t"
    assert payload["display_limit_ci"] == result.display_limit_ci
    assert payload["warnings"] == result.warnings
    assert payload["uses_continuity_correction"] is False
    assert result.display_limit_estimate == pytest.approx(
        math.exp(result.limit_estimate)
    )
    assert result.display_limit_ci == pytest.approx(
        tuple(math.exp(value) for value in result.limit_ci)
    )
    assert "Harbord efficient-score test" in str(result)
    assert "multiplicative dispersion" in str(result)
