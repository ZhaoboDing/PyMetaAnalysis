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
    (REFERENCE_DIR / "peters_small_study_effects_meta.json").read_text(encoding="utf-8")
)
# R's qt() and older SciPy t.ppf() differ by roughly 2e-11 in the critical
# value at df=10. Endpoint tolerances scale that implementation difference by
# the corresponding coefficient standard error; coefficient checks stay tight.
R_T_SLOPE_CI_ATOL = 3e-9
R_T_LIMIT_CI_ATOL = 2e-11


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


def test_peters_matches_direct_weighted_regression_calculation() -> None:
    source = _binary_result()
    studies = source.study_results.loc[source.study_results["included"]]
    n = (studies["n_treat"] + studies["n_control"]).to_numpy(dtype=float)
    events = (studies["event_treat"] + studies["event_control"]).to_numpy(dtype=float)
    weights = events * (n - events) / n
    design = np.column_stack((np.ones(len(studies)), 1.0 / n))
    response = studies["effect"].to_numpy(dtype=float)
    information = design.T @ (weights[:, np.newaxis] * design)
    coefficients = np.linalg.solve(information, design.T @ (weights * response))
    residual = response - design @ coefficients
    dispersion = float(np.dot(weights, residual * residual) / (len(studies) - 2))
    covariance = dispersion * np.linalg.inv(information)
    coefficient_se = np.sqrt(np.diag(covariance))

    result = source.peters_test()

    assert result.limit_estimate == pytest.approx(coefficients[0], rel=2e-14)
    assert result.slope == pytest.approx(coefficients[1], rel=2e-14)
    assert result.limit_standard_error == pytest.approx(coefficient_se[0], rel=2e-14)
    assert result.slope_standard_error == pytest.approx(coefficient_se[1], rel=2e-14)
    assert result.statistic == pytest.approx(
        coefficients[1] / coefficient_se[1], rel=2e-14
    )
    assert result.residual_dispersion == pytest.approx(dispersion, rel=2e-14)


def test_peters_matches_meta_reference() -> None:
    result = _binary_result().peters_test()

    assert isinstance(result, ma.PetersTestResult)
    assert REFERENCE["generated_by"] == "R meta"
    assert REFERENCE["meta_version"] == "8.5.0"
    assert result.k == REFERENCE["k"]
    assert result.confidence_level == pytest.approx(REFERENCE["confidence_level"])
    assert result.slope == pytest.approx(REFERENCE["slope"], rel=5e-13)
    assert result.slope_standard_error == pytest.approx(
        REFERENCE["slope_standard_error"], rel=5e-13
    )
    assert result.slope_ci == pytest.approx(
        REFERENCE["slope_ci"], rel=5e-13, abs=R_T_SLOPE_CI_ATOL
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
        REFERENCE["limit_ci"], rel=5e-13, abs=R_T_LIMIT_CI_ATOL
    )
    assert result.residual_dispersion == pytest.approx(
        REFERENCE["residual_dispersion"], rel=5e-13
    )
    assert result.continuity_correction == REFERENCE["continuity_correction"]
    assert result.correction_scope == "only_zero_studies"
    assert result.corrected_studies == REFERENCE["corrected_studies"]
    assert result.method == "peters"
    assert result.model == "weighted_regression_multiplicative_dispersion"
    assert result.predictor == "inverse_total_sample_size"
    assert result.weight_method == "S_times_F_over_N"
    assert np.isfinite(result.condition_number)


@pytest.mark.parametrize(
    ("method", "model"),
    [("MH", "common"), ("IV", "common"), ("IV", "random"), ("Peto", "common")],
)
def test_peters_is_independent_of_source_pooling_method(
    method: str, model: str
) -> None:
    expected = _binary_result().peters_test()
    result = _binary_result(method=method, model=model).peters_test()

    assert result.slope == pytest.approx(expected.slope, rel=2e-14)
    assert result.slope_standard_error == pytest.approx(
        expected.slope_standard_error, rel=2e-14
    )
    assert result.limit_estimate == pytest.approx(expected.limit_estimate, rel=2e-14)
    assert result.pvalue == pytest.approx(expected.pvalue, rel=2e-14)
    if method == "Peto":
        assert any("Peto pooling" in warning for warning in result.warnings)


def test_peters_is_invariant_to_row_order_and_tracks_source_correction() -> None:
    expected = _binary_result().peters_test()
    reordered = _binary_result(
        REFERENCE_DATA.iloc[::-1].reset_index(drop=True)
    ).peters_test()
    stronger_correction = _binary_result(
        continuity_correction=1.0,
        correction_scope="all_studies",
    ).peters_test()

    assert reordered.slope == pytest.approx(expected.slope, rel=2e-14)
    assert reordered.limit_estimate == pytest.approx(expected.limit_estimate, rel=2e-14)
    assert stronger_correction.continuity_correction == 1.0
    assert stronger_correction.correction_scope == "all_studies"
    assert stronger_correction.corrected_studies == len(REFERENCE_DATA)
    assert stronger_correction.slope != pytest.approx(expected.slope)


def test_peters_ignores_excluded_rows() -> None:
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

    result = _binary_result(data).peters_test()
    expected = _binary_result().peters_test()

    assert result.k == len(REFERENCE_DATA)
    assert result.slope == pytest.approx(expected.slope, rel=2e-14)
    assert result.limit_estimate == pytest.approx(expected.limit_estimate, rel=2e-14)


def test_peters_changes_sign_when_treatment_and_control_are_swapped() -> None:
    swapped = REFERENCE_DATA.rename(
        columns={
            "event_treat": "event_control",
            "n_treat": "n_control",
            "event_control": "event_treat",
            "n_control": "n_treat",
        }
    )
    original = _binary_result().peters_test()
    reversed_result = _binary_result(swapped).peters_test()

    assert reversed_result.slope == pytest.approx(-original.slope, rel=2e-14)
    assert reversed_result.statistic == pytest.approx(-original.statistic, rel=2e-14)
    assert reversed_result.pvalue == pytest.approx(original.pvalue, rel=2e-14)
    assert reversed_result.limit_estimate == pytest.approx(
        -original.limit_estimate, rel=2e-14
    )
    assert reversed_result.display_limit_estimate == pytest.approx(
        1.0 / original.display_limit_estimate, rel=2e-14
    )


def test_peters_reuses_or_overrides_fitted_confidence_level() -> None:
    fitted = _binary_result(confidence_level=0.9)

    inherited = fitted.peters_test()
    overridden = fitted.peters_test(confidence_level=0.8)

    assert inherited.confidence_level == 0.9
    assert overridden.confidence_level == 0.8
    assert overridden.slope_ci_low > inherited.slope_ci_low
    assert overridden.slope_ci_high < inherited.slope_ci_high
    assert overridden.limit_ci_low > inherited.limit_ci_low
    assert overridden.limit_ci_high < inherited.limit_ci_high


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, np.nan, np.inf, True, "0.95"])
def test_peters_rejects_invalid_confidence_level(confidence_level: object) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="between 0 and 1"):
        _binary_result().peters_test(  # type: ignore[arg-type]
            confidence_level=confidence_level
        )


def test_peters_requires_binary_odds_ratios_with_retained_counts() -> None:
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
        risk_ratio.peters_test()
    with pytest.raises(ma.UnsupportedMethodError, match=r"meta_binary\(\)"):
        generic.peters_test()


def test_peters_requires_three_studies_and_varying_total_sizes() -> None:
    two_studies = _binary_result(REFERENCE_DATA.iloc[:2].copy())
    equal_sizes = pd.DataFrame(
        {
            "study": ["A", "B", "C", "D"],
            "event_treat": [4, 6, 8, 10],
            "n_treat": [50, 50, 50, 50],
            "event_control": [7, 9, 5, 12],
            "n_control": [50, 50, 50, 50],
        }
    )

    with pytest.raises(ma.InsufficientStudiesError, match="at least three"):
        two_studies.peters_test()
    with pytest.raises(ma.InvalidStudyDataError, match="total sample sizes"):
        _binary_result(equal_sizes).peters_test()


def test_peters_rejects_uncorrected_zero_cells() -> None:
    source = _binary_result(
        method="Peto",
        continuity_correction=0.0,
        correction_scope="none",
    )

    with pytest.raises(ma.InvalidStudyDataError, match="positive four-cell counts"):
        source.peters_test()


def test_peters_records_few_study_and_interpretation_warnings() -> None:
    result = _binary_result(REFERENCE_DATA.iloc[:5].copy()).peters_test()

    assert any("fewer than ten" in warning for warning in result.warnings)
    assert any("diagnostic-accuracy" in warning for warning in result.warnings)
    assert any(
        "does not by itself establish or exclude publication bias" in warning
        for warning in result.warnings
    )


def test_peters_result_is_immutable_printable_and_machine_readable() -> None:
    result = _binary_result().peters_test()
    payload = result.to_dict()

    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0.5  # type: ignore[misc]
    assert payload["statistic_name"] == "t"
    assert payload["distribution"] == "t"
    assert payload["display_limit_ci"] == result.display_limit_ci
    assert payload["warnings"] == result.warnings
    assert result.display_limit_estimate == pytest.approx(
        math.exp(result.limit_estimate)
    )
    assert result.display_limit_ci == pytest.approx(
        tuple(math.exp(value) for value in result.limit_ci)
    )
    assert "Peters regression test for funnel-plot asymmetry" in str(result)
    assert "weighted regression with multiplicative dispersion" in str(result)
