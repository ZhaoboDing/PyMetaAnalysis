"""Regression values cross-checked against independent implementations."""

from __future__ import annotations

import numpy as np
import pytest

import meta_analyze as ma

# These reference values were generated with statsmodels 0.14.6
# statsmodels.stats.meta_analysis.combine_effects. Statsmodels documents these
# methods as independently verified against R meta and metafor.
EFFECT = np.array([-0.4, 0.1, 0.5, 1.2, 1.8], dtype=float)
VARIANCE = np.array([0.04, 0.09, 0.05, 0.16, 0.08], dtype=float)


def test_common_and_dl_match_statsmodels_reference() -> None:
    common = ma.meta_analysis(effect=EFFECT, variance=VARIANCE, model="common")
    random = ma.meta_analysis(
        effect=EFFECT,
        variance=VARIANCE,
        model="random",
        tau2_method="DL",
    )

    assert common.estimate == pytest.approx(0.4155844155844156, abs=1e-13)
    assert common.standard_error == pytest.approx(0.11557711927941293, abs=1e-13)
    assert common.q == pytest.approx(45.68181818181819, abs=1e-12)
    assert random.tau2 == pytest.approx(0.7324042379788102, abs=1e-12)
    assert random.estimate == pytest.approx(0.6193796045896693, abs=1e-12)
    assert random.standard_error == pytest.approx(0.4035605730212698, abs=1e-12)


def test_pm_matches_statsmodels_reference() -> None:
    result = ma.meta_analysis(
        effect=EFFECT,
        variance=VARIANCE,
        model="random",
        tau2_method="PM",
    )

    assert result.tau2 == pytest.approx(0.6891203494172782, abs=1e-9)
    assert result.estimate == pytest.approx(0.6182360899515967, abs=1e-10)
    assert result.standard_error == pytest.approx(0.3926598958798601, abs=1e-10)


def test_reml_equal_variance_case_has_closed_form_solution() -> None:
    effect = np.array([0.0, 1.0, 2.0])
    within_study_variance = 0.25
    expected_tau2 = np.var(effect, ddof=1) - within_study_variance

    result = ma.meta_analysis(
        effect=effect,
        variance=np.repeat(within_study_variance, len(effect)),
        model="random",
        tau2_method="REML",
    )

    assert result.tau2 == pytest.approx(expected_tau2, abs=1e-10)
