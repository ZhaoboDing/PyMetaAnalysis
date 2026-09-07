"""Core inference audit: pinned software comparisons and independent oracles."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import chi2, t

import meta_analyze as ma
from meta_analyze.estimators.inverse_variance import _confidence_interval
from meta_analyze.heterogeneity import tau2_inconsistency

REFERENCE = json.loads(
    (Path(__file__).parent / "reference/core_inference_metafor.json").read_text(
        encoding="utf-8"
    )
)
CASES = REFERENCE["cases"]


def test_core_review_reference_environment_and_scope() -> None:
    assert REFERENCE["r_version"] == "4.6.1"
    assert REFERENCE["metafor_version"] == "5.0.1"
    assert REFERENCE["jsonlite_version"] == "2.0.0"
    assert len(CASES) == 24
    assert {case["k"] for case in CASES} == {2, 3, 5, 10}
    assert all(len(case["fits"]) == 9 for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("fit_index", range(9))
def test_core_inference_matches_pinned_metafor(
    case: dict[str, Any], fit_index: int
) -> None:
    expected = case["fits"][fit_index]
    result = ma.meta_analysis(
        effect=case["effect"],
        variance=case["variance"],
        tau2_method=expected["tau2_method"],
        ci_method=expected["ci_method"],
        atol=1e-12,
    )
    # Iterative fits include solver error; DL is closed form. PI multiplies
    # SE error by t_(k-2), so its absolute bound is separately stated below.
    rtol, atol = (5e-13, 5e-15) if expected["tau2_method"] == "DL" else (2e-10, 2e-11)
    np.testing.assert_allclose(
        [result.estimate, result.standard_error, *result.ci, result.tau2],
        [
            expected["estimate"],
            expected["standard_error"],
            *expected["ci"],
            expected["tau2"],
        ],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        result.study_results["normalized_weight"],
        expected["weights"],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        [result.q, result.heterogeneity.pvalue],
        [expected["q"], expected["q_p_value"]],
        rtol=5e-13,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        [result.i2, result.h2], [expected["i2"], expected["h2"]], rtol=rtol, atol=atol
    )
    if case["k"] == 2:
        assert result.prediction_interval is None
        assert result.method.prediction_interval_method is None
    else:
        np.testing.assert_allclose(
            result.prediction_interval,
            expected["prediction_interval"],
            rtol=2e-9,
            atol=1e-10,
        )
        assert result.method.prediction_interval_method == (
            "HTS" if expected["ci_method"] == "normal" else "HK-PR"
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_core_common_and_q_profile_match_metafor(case: dict[str, Any]) -> None:
    common = ma.meta_analysis(
        effect=case["effect"], variance=case["variance"], model="common"
    )
    expected = case["common"]
    np.testing.assert_allclose(
        [common.estimate, common.standard_error, *common.ci, common.i2, common.h2],
        [
            expected["estimate"],
            expected["standard_error"],
            *expected["ci"],
            expected["i2"],
            expected["h2"],
        ],
        rtol=5e-13,
        atol=5e-15,
    )
    result = ma.meta_analysis(
        effect=case["effect"], variance=case["variance"], atol=1e-12
    )
    interval = result.tau2_confidence_interval(atol=1e-12)
    np.testing.assert_allclose(
        interval.ci, case["q_profile"]["tau2_ci"], rtol=2e-9, atol=1e-10
    )
    assert interval.is_empty == case["q_profile"]["empty_by_q_at_zero"]
    assert interval.is_empty == case["q_profile"]["native_empty"]
    assert case["q_profile"]["upper_sign"] != ">"


@pytest.mark.parametrize("method", ["DL", "PM", "REML"])
@pytest.mark.parametrize("ci_method", ["hartung_knapp", "hartung_knapp_adhoc"])
@pytest.mark.parametrize("shift", [0.0, 1e16, -1e16])
@pytest.mark.parametrize(
    "effects, variance", [([0, 2], 4), ([0, 2], 1), ([0, 2, 6], 16)]
)
def test_hk_large_location_matches_exact_rational_variance(
    method: str, ci_method: str, shift: float, effects: list[int], variance: int
) -> None:
    k = len(effects)
    mean = sum(map(Fraction, effects)) / k
    sample_variance = sum((Fraction(y) - mean) ** 2 for y in effects) / (k - 1)
    tau2 = max(Fraction(0), sample_variance - variance)
    expected_variance = sample_variance / k
    if ci_method == "hartung_knapp_adhoc":
        expected_variance = max(expected_variance, (variance + tau2) / k)
    # All input differences survive the shift exactly; only the rounded pooled
    # location loses the fractional mean. Its rounding must not enter the SE.
    result = ma.meta_analysis(
        effect=np.asarray(effects) + shift,
        variance=[variance] * k,
        tau2_method=method,
        ci_method=ci_method,
    )
    assert result.tau2 == pytest.approx(float(tau2), abs=1e-12)
    assert result.standard_error == pytest.approx(
        float(expected_variance) ** 0.5, rel=2e-15
    )


def test_hk_weights_are_applied_before_squaring_large_residuals() -> None:
    # Isolate CI arithmetic from tau estimation: residual^2 overflows even
    # though its precision-weighted contribution and resulting SE are finite.
    with np.errstate(over="raise", invalid="raise"):
        _, _, se, _ = _confidence_interval(
            effect=np.array([0.0, 1e160]),
            weights=np.array([1.0, 1e-30]),
            estimate=1e130,
            classic_variance=1.0,
            ci_method="hartung_knapp",
            confidence_level=0.95,
        )
    assert se == pytest.approx(1e145, rel=2e-15)


@pytest.mark.parametrize("k", [3, 5, 10])
def test_inconsistency_preserves_finite_typical_variance_near_float_limit(
    k: int,
) -> None:
    with np.errstate(over="raise", invalid="raise"):
        i2, h2 = tau2_inconsistency(np.full(k, 1e308), 1e307)
    assert i2 == pytest.approx(1 / 11, rel=2e-15)
    assert h2 == pytest.approx(1.1, rel=2e-15)


@pytest.mark.parametrize("k", [2, 3, 5, 10])
@pytest.mark.parametrize("spread", [0.0, 0.01, 0.5, 2.0])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_equal_variance_q_profile_and_prediction_have_closed_form_oracles(
    k: int, spread: float, scale: float
) -> None:
    effects = np.linspace(-spread, spread, k) * scale
    variance = 0.2 * scale**2
    ss = float(effects @ effects)
    targets = chi2.ppf([0.975, 0.025], df=k - 1)
    unconstrained = ss / targets - variance
    result = ma.meta_analysis(
        effect=effects,
        variance=[variance] * k,
        tau2_method="DL",
        ci_method="hartung_knapp",
    )
    interval = result.tau2_confidence_interval(atol=1e-12 * min(1.0, scale**2))
    np.testing.assert_allclose(
        interval.ci, np.maximum(0.0, unconstrained), rtol=2e-10, atol=1e-12 * scale**2
    )
    assert interval.is_empty == (unconstrained[1] < 0)
    if k >= 3:
        mean_variance = ss / ((k - 1) * k)
        tau2 = max(0.0, ss / (k - 1) - variance)
        margin = t.ppf(0.975, k - 2) * np.sqrt(tau2 + mean_variance)
        np.testing.assert_allclose(
            result.prediction_interval, [-margin, margin], rtol=2e-14, atol=1e-15
        )
