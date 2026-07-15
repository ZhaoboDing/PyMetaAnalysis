"""Cross-software regression tests against committed R/metafor results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
GENERIC_DATA = pd.read_csv(REFERENCE_DIR / "generic_input.csv")
BINARY_DATA = pd.read_csv(REFERENCE_DIR / "binary_input.csv")
SPARSE_BINARY_DATA = pd.read_csv(REFERENCE_DIR / "binary_sparse_input.csv")


def _load_reference(filename: str) -> dict[str, Any]:
    return json.loads((REFERENCE_DIR / filename).read_text(encoding="utf-8"))


GENERIC = _load_reference("generic_metafor.json")
BINARY = _load_reference("binary_metafor.json")
CONTINUOUS = _load_reference("continuous_metafor.json")

CLOSED_RTOL = 5e-13
CLOSED_ATOL = 5e-15
ITERATIVE_RTOL = 2e-10
ITERATIVE_ATOL = 2e-11


def _binary_columns() -> dict[str, str]:
    return {
        "event_treat": "event_treat",
        "n_treat": "n_treat",
        "event_control": "event_control",
        "n_control": "n_control",
    }


def _assert_fit(
    result: ma.MetaAnalysisResult,
    expected: dict[str, Any],
    *,
    iterative: bool = False,
) -> None:
    rtol = ITERATIVE_RTOL if iterative else CLOSED_RTOL
    atol = ITERATIVE_ATOL if iterative else CLOSED_ATOL
    actual = [
        result.estimate,
        result.standard_error,
        result.ci_low,
        result.ci_high,
        result.tau2,
    ]
    reference = [
        expected["estimate"],
        expected["standard_error"],
        *expected["ci"],
        expected.get("tau2", 0.0),
    ]
    np.testing.assert_allclose(actual, reference, rtol=rtol, atol=atol)

    studies = result.study_results
    included_weights = studies.loc[studies["included"], "normalized_weight"]
    np.testing.assert_allclose(
        included_weights,
        expected["weights"],
        rtol=rtol,
        atol=atol,
    )
    assert included_weights.sum() == pytest.approx(1.0, abs=2e-15)


def test_reference_fixtures_record_a_consistent_r_environment() -> None:
    fixtures = [GENERIC, BINARY, CONTINUOUS]

    assert {fixture["generated_by"] for fixture in fixtures} == {"R metafor"}
    assert len({fixture["r_version"] for fixture in fixtures}) == 1
    assert len({fixture["metafor_version"] for fixture in fixtures}) == 1
    assert len({fixture["jsonlite_version"] for fixture in fixtures}) == 1
    assert GENERIC["iterative_control"] == {
        "tolerance": 1e-10,
        "max_iterations": 1000,
    }
    assert BINARY["iterative_control"] == GENERIC["iterative_control"]


def test_generic_heterogeneity_matches_metafor() -> None:
    expected = GENERIC["heterogeneity"]
    result = ma.meta_analysis(
        GENERIC_DATA,
        effect="effect",
        variance="variance",
        study="study",
        model="common",
    )

    np.testing.assert_allclose(
        [result.q, result.q_pvalue, result.i2, result.h2],
        [expected["q"], expected["pvalue"], expected["i2"], expected["h2"]],
        rtol=5e-12,
        atol=5e-15,
    )
    assert result.q_df == expected["df"]


@pytest.mark.parametrize("tau2_method", ["DL", "PM", "REML"])
def test_generic_normal_interval_models_match_metafor(tau2_method: str) -> None:
    result = ma.meta_analysis(
        GENERIC_DATA,
        effect="effect",
        variance="variance",
        study="study",
        model="random",
        tau2_method=tau2_method,
    )

    _assert_fit(
        result,
        GENERIC["random"][tau2_method],
        iterative=tau2_method in {"PM", "REML"},
    )


def test_generic_common_effect_model_matches_metafor() -> None:
    result = ma.meta_analysis(
        GENERIC_DATA,
        effect="effect",
        variance="variance",
        study="study",
        model="common",
    )

    _assert_fit(result, GENERIC["common"])


@pytest.mark.parametrize(
    ("ci_method", "reference_key"),
    [
        ("hartung_knapp", "reml_hartung_knapp"),
        ("hartung_knapp_adhoc", "reml_hartung_knapp_adhoc"),
    ],
)
def test_generic_hartung_knapp_intervals_match_metafor(
    ci_method: str,
    reference_key: str,
) -> None:
    result = ma.meta_analysis(
        GENERIC_DATA,
        effect="effect",
        variance="variance",
        study="study",
        model="random",
        tau2_method="REML",
        ci_method=ci_method,
    )

    _assert_fit(result, GENERIC[reference_key], iterative=True)


def test_generic_hts_prediction_interval_matches_metafor_formula() -> None:
    result = ma.meta_analysis(
        GENERIC_DATA,
        effect="effect",
        variance="variance",
        study="study",
        model="random",
        tau2_method="REML",
    )

    assert result.prediction_interval is not None
    np.testing.assert_allclose(
        result.prediction_interval,
        GENERIC["reml_prediction_interval_hts"],
        rtol=ITERATIVE_RTOL,
        atol=ITERATIVE_ATOL,
    )


@pytest.mark.parametrize("measure", ["OR", "RR", "RD"])
def test_clean_binary_inverse_variance_models_match_metafor(measure: str) -> None:
    expected = BINARY["clean"][measure]
    common = ma.meta_binary(
        BINARY_DATA,
        **_binary_columns(),
        study="study",
        measure=measure,
        method="IV",
        model="common",
    )
    random = ma.meta_binary(
        BINARY_DATA,
        **_binary_columns(),
        study="study",
        measure=measure,
        method="IV",
        model="random",
        tau2_method="REML",
    )

    np.testing.assert_allclose(
        common.study_results["effect"],
        expected["effect"],
        rtol=CLOSED_RTOL,
        atol=CLOSED_ATOL,
    )
    np.testing.assert_allclose(
        common.study_results["variance"],
        expected["variance"],
        rtol=CLOSED_RTOL,
        atol=CLOSED_ATOL,
    )
    _assert_fit(common, expected["common_iv"])
    _assert_fit(random, expected["random_reml_iv"], iterative=True)


@pytest.mark.parametrize("measure", ["OR", "RR"])
def test_clean_binary_mantel_haenszel_matches_metafor(measure: str) -> None:
    result = ma.meta_binary(
        BINARY_DATA,
        **_binary_columns(),
        study="study",
        measure=measure,
        method="MH",
    )

    _assert_fit(result, BINARY["clean"][measure]["mantel_haenszel"])
    assert not result.study_results["mh_continuity_corrected"].any()


@pytest.mark.parametrize("measure", ["OR", "RR"])
@pytest.mark.parametrize(
    ("method", "reference_key"),
    [("IV", "common_iv"), ("MH", "mantel_haenszel")],
)
def test_sparse_binary_zero_event_handling_matches_metafor(
    measure: str,
    method: str,
    reference_key: str,
) -> None:
    expected = BINARY["sparse"][measure]
    result = ma.meta_binary(
        SPARSE_BINARY_DATA,
        **_binary_columns(),
        study="study",
        measure=measure,
        method=method,
        model="common",
    )
    studies = result.study_results

    np.testing.assert_allclose(
        studies["effect"],
        np.asarray(expected["effect"], dtype=np.float64),
        rtol=CLOSED_RTOL,
        atol=CLOSED_ATOL,
    )
    np.testing.assert_allclose(
        studies["variance"],
        np.asarray(expected["variance"], dtype=np.float64),
        rtol=CLOSED_RTOL,
        atol=CLOSED_ATOL,
    )
    assert studies["included"].tolist() == [True, True, False, False, True]
    assert studies.loc[2, "exclusion_reason"] == "no events in either group"
    assert (
        studies.loc[3, "exclusion_reason"]
        == "all participants have events in both groups"
    )
    assert studies.loc[~studies["included"], "normalized_weight"].isna().all()
    _assert_fit(result, expected[reference_key])
