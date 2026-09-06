from __future__ import annotations

import itertools
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_DATA = pd.read_csv(REFERENCE_DIR / "small_study_effects_input.csv")
REFERENCE = json.loads(
    (REFERENCE_DIR / "begg_ranktest_metafor.json").read_text(encoding="utf-8")
)


def _reference_result(**kwargs: object) -> ma.MetaAnalysisResult:
    options: dict[str, object] = {
        "effect": "effect",
        "variance": "variance",
        "study": "study",
        "model": "common",
    }
    options.update(kwargs)
    return ma.meta_analysis(REFERENCE_DATA, **options)  # type: ignore[arg-type]


def _kendall_s(x: np.ndarray, y: np.ndarray) -> int:
    statistic = 0
    for first, second in itertools.combinations(range(len(x)), 2):
        statistic += int(np.sign((x[first] - x[second]) * (y[first] - y[second])))
    return statistic


def test_begg_matches_exact_permutation_calculation() -> None:
    effect = np.asarray([-0.4, 0.2, 0.8, 1.1, 1.7])
    variance = np.asarray([0.09, 0.01, 0.16, 0.04, 0.25])
    weights = 1 / variance
    center = np.average(effect, weights=weights)
    pooled_variance = 1 / np.sum(weights)
    response = (effect - center) / np.sqrt(variance - pooled_variance)
    observed = _kendall_s(response, variance)
    permutation_statistics = [
        _kendall_s(response, np.asarray(permutation))
        for permutation in itertools.permutations(variance)
    ]
    expected_pvalue = np.mean(np.abs(permutation_statistics) >= abs(observed))

    result = ma.meta_analysis(effect=effect, variance=variance).begg_test(exact=True)

    assert result.tau == pytest.approx(observed / 10)
    assert result.statistic == observed
    assert result.pvalue == pytest.approx(expected_pvalue, abs=1e-15)
    assert result.statistic_name == "Kendall S"
    assert result.distribution == "exact"
    assert result.inference_method == "exact"


def test_begg_matches_metafor_exact_reference() -> None:
    result = _reference_result().begg_test()

    assert isinstance(result, ma.BeggTestResult)
    assert result.k == REFERENCE["k"]
    assert result.tau == pytest.approx(REFERENCE["tau"], abs=1e-15)
    assert result.statistic == REFERENCE["statistic"]
    assert result.pvalue == pytest.approx(REFERENCE["pvalue"], abs=1e-15)
    assert result.correlation_method == REFERENCE["correlation_method"]
    assert result.inference_method == REFERENCE["inference_method"]
    assert result.response_tied_pairs == 0
    assert result.variance_tied_pairs == 0
    assert result.joint_tied_pairs == 0


def test_begg_result_is_immutable_auditable_and_printable() -> None:
    result = _reference_result().begg_test()
    payload = result.to_dict()

    assert payload["response"] == "standardized centered effect"
    assert payload["predictor"] == "sampling variance"
    assert payload["studies"] == 12
    assert any("publication bias" in note for note in payload["warnings"])
    assert "Begg-Mazumdar rank-correlation test" in str(result)
    assert "Kendall's tau-b" in str(result)
    with pytest.raises(FrozenInstanceError):
        result.tau = 0.0  # type: ignore[misc]
    replaced = replace(result, pvalue=0.5)
    assert replaced.pvalue == 0.5


def test_begg_matches_metafor_tied_asymptotic_reference() -> None:
    data = REFERENCE_DATA.copy()
    data.loc[1, "variance"] = data.loc[0, "variance"]
    result = ma.meta_analysis(
        data,
        effect="effect",
        variance="variance",
        study="study",
    ).begg_test()
    expected = REFERENCE["tied_asymptotic"]

    assert result.inference_method == "asymptotic"
    assert result.tau == pytest.approx(expected["tau"], abs=1e-15)
    assert result.statistic == pytest.approx(expected["statistic"], abs=1e-15)
    assert result.pvalue == pytest.approx(expected["pvalue"], abs=1e-15)
    assert result.variance_tied_pairs == 1

    corrected = ma.meta_analysis(
        data,
        effect="effect",
        variance="variance",
        study="study",
    ).begg_test(continuity_correction=True)
    expected_corrected = REFERENCE["tied_continuity_corrected"]
    assert corrected.statistic == pytest.approx(
        expected_corrected["statistic"], abs=1e-15
    )
    assert corrected.pvalue == pytest.approx(expected_corrected["pvalue"], abs=1e-15)


def test_asymptotic_tie_adjustment_matches_direct_calculation() -> None:
    effect = np.asarray([0.1, 0.1, 0.4, 0.8, 1.2, 1.5])
    variance = np.asarray([0.01, 0.04, 0.04, 0.09, 0.16, 0.25])
    weights = 1 / variance
    center = np.average(effect, weights=weights)
    pooled_variance = 1 / np.sum(weights)
    response = (effect - center) / np.sqrt(variance - pooled_variance)
    kendall_s = _kendall_s(response, variance)
    n = len(effect)
    pairs = n * (n - 1) // 2
    response_ties = 0
    variance_ties = 1
    expected_tau = kendall_s / math.sqrt(
        (pairs - response_ties) * (pairs - variance_ties)
    )
    m = n * (n - 1)
    variance_tie_moment = 2 * 1 * (2 * 2 + 5)
    variance_s = (
        m * (2 * n + 5) - variance_tie_moment
    ) / 18 + 2 * response_ties * variance_ties / m
    expected_z = kendall_s / math.sqrt(variance_s)

    result = ma.meta_analysis(effect=effect, variance=variance).begg_test()

    assert result.tau == pytest.approx(expected_tau, abs=1e-15)
    assert result.statistic == pytest.approx(expected_z, abs=1e-15)
    assert result.pvalue == pytest.approx(2 * norm.sf(abs(expected_z)), abs=1e-15)
    assert result.inference_method == "asymptotic"
    assert result.response_tied_pairs == 0
    assert result.variance_tied_pairs == 1


def test_asymptotic_continuity_correction_moves_statistic_toward_zero() -> None:
    source = _reference_result()
    uncorrected = source.begg_test(exact=False)
    corrected = source.begg_test(
        exact=False,
        continuity_correction=True,
    )

    assert corrected.tau == uncorrected.tau
    assert abs(corrected.statistic) < abs(uncorrected.statistic)
    assert corrected.pvalue > uncorrected.pvalue
    assert corrected.continuity_correction is True


def test_asymptotic_continuity_correction_handles_negative_association() -> None:
    source = ma.meta_analysis(
        effect=[0.1, 0.4, 0.8, 1.2, 1.7],
        variance=[0.25, 0.16, 0.09, 0.04, 0.01],
    )
    uncorrected = source.begg_test(exact=False)
    corrected = source.begg_test(exact=False, continuity_correction=True)

    assert uncorrected.statistic < 0
    assert corrected.statistic < 0
    assert abs(corrected.statistic) < abs(uncorrected.statistic)


def test_auto_inference_uses_asymptotic_when_correction_is_requested() -> None:
    result = _reference_result().begg_test(continuity_correction=True)

    assert result.inference_method == "asymptotic"
    assert result.continuity_correction is True


def test_auto_inference_uses_asymptotic_at_fifty_studies() -> None:
    result = ma.meta_analysis(
        effect=np.arange(50, dtype=float),
        variance=np.square(np.arange(1, 51, dtype=float)),
    ).begg_test()

    assert result.inference_method == "asymptotic"
    assert any("publication bias" in note for note in result.warnings)


def test_joint_ties_are_reported_separately() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.1, 0.4, 0.8],
        variance=[0.01, 0.01, 0.04, 0.09],
    ).begg_test()

    assert result.response_tied_pairs == 1
    assert result.variance_tied_pairs == 1
    assert result.joint_tied_pairs == 1


@pytest.mark.parametrize(
    ("model", "tau2_method"),
    [
        ("common", None),
        ("random", "DL"),
        ("random", "PM"),
        ("random", "REML"),
    ],
)
def test_begg_is_independent_of_source_model(
    model: str,
    tau2_method: str | None,
) -> None:
    expected = _reference_result().begg_test()
    result = _reference_result(model=model, tau2_method=tau2_method).begg_test()

    assert result.tau == expected.tau
    assert result.statistic == expected.statistic
    assert result.pvalue == expected.pvalue


def test_begg_is_invariant_to_order_location_and_variance_scale() -> None:
    expected = _reference_result().begg_test()
    transformed = REFERENCE_DATA.sample(frac=1.0, random_state=17).copy()
    transformed["effect"] += 500.0
    transformed["variance"] *= 1e120
    result = ma.meta_analysis(
        transformed,
        effect="effect",
        variance="variance",
    ).begg_test()

    assert result.tau == expected.tau
    assert result.statistic == expected.statistic
    assert result.pvalue == expected.pvalue


def test_begg_uses_only_included_studies_and_does_not_mutate_source() -> None:
    data = REFERENCE_DATA.copy()
    data.loc[len(data)] = ["Excluded", np.nan, 0.04]
    source = ma.meta_analysis(
        data,
        effect="effect",
        variance="variance",
        study="study",
        missing="drop",
    )
    before = source.study_results

    result = source.begg_test()

    assert result.tau == _reference_result().begg_test().tau
    pd.testing.assert_frame_equal(source.study_results, before)


@pytest.mark.parametrize("exact", [0, 1, "auto", 1.0])
def test_invalid_exact_is_rejected(exact: object) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="exact"):
        _reference_result().begg_test(exact=exact)  # type: ignore[arg-type]


@pytest.mark.parametrize("correction", [0, 1, "yes", None])
def test_invalid_continuity_correction_is_rejected(correction: object) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="continuity_correction"):
        _reference_result().begg_test(  # type: ignore[arg-type]
            continuity_correction=correction
        )


def test_exact_rejects_ties_and_continuity_correction() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="ties"):
        ma.meta_analysis(
            effect=[0.1, 0.2, 0.5],
            variance=[0.01, 0.01, 0.09],
        ).begg_test(exact=True)
    with pytest.raises(ma.InvalidStudyDataError, match="asymptotic"):
        _reference_result().begg_test(exact=True, continuity_correction=True)


def test_too_few_and_constant_inputs_are_rejected() -> None:
    with pytest.raises(ma.InsufficientStudiesError, match="three"):
        ma.meta_analysis(effect=[0.1, 0.2], variance=[0.01, 0.04]).begg_test()
    with pytest.raises(ma.InvalidStudyDataError, match="variation"):
        ma.meta_analysis(
            effect=[0.1, 0.1, 0.1],
            variance=[0.01, 0.04, 0.09],
        ).begg_test()
    with pytest.raises(ma.InvalidStudyDataError, match="variation"):
        ma.meta_analysis(
            effect=[0.1, 0.2, 0.3],
            variance=[0.04, 0.04, 0.04],
        ).begg_test()


def test_small_analysis_warns_and_result_is_immutable_and_serializable() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.4, 0.8, 1.2],
        variance=[0.01, 0.09, 0.04, 0.16],
    ).begg_test()

    assert any("fewer than ten" in warning for warning in result.warnings)
    with pytest.raises(FrozenInstanceError):
        result.tau = 0.0  # type: ignore[misc]
    assert replace(result, tau=0.0).tau == 0.0
    payload = result.to_dict()
    assert payload["tau"] == result.tau
    assert payload["studies"] == result.k
    assert payload["warnings"] == result.warnings
    assert "Begg-Mazumdar rank-correlation test" in str(result)
    assert "Kendall's tau-b" in str(result)
