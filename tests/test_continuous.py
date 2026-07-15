"""Continuous meta-analysis tests with metafor reference values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
DATA = pd.read_csv(REFERENCE_DIR / "continuous_input.csv")
REFERENCE: dict[str, Any] = json.loads(
    (REFERENCE_DIR / "continuous_metafor.json").read_text(encoding="utf-8")
)


def _columns() -> dict[str, str]:
    return {
        "mean_treat": "mean_treat",
        "sd_treat": "sd_treat",
        "n_treat": "n_treat",
        "mean_control": "mean_control",
        "sd_control": "sd_control",
        "n_control": "n_control",
    }


@pytest.mark.parametrize("measure", ["MD", "SMD"])
def test_continuous_common_effect_matches_metafor_reference(measure: str) -> None:
    expected = REFERENCE[measure.lower()]
    result = ma.meta_continuous(
        DATA,
        **_columns(),
        study="study",
        measure=measure,
        model="common",
    )

    np.testing.assert_allclose(
        result.study_results["effect"], expected["effect"], rtol=2e-14, atol=1e-15
    )
    np.testing.assert_allclose(
        result.study_results["variance"],
        expected["variance"],
        rtol=2e-14,
        atol=1e-15,
    )
    assert result.estimate == pytest.approx(expected["estimate"], rel=2e-14)
    assert result.standard_error == pytest.approx(expected["standard_error"], rel=2e-14)
    assert result.ci == pytest.approx(expected["ci"], rel=2e-14)
    assert result.study_results["normalized_weight"].sum() == pytest.approx(1.0)


def test_smd_records_exact_hedges_g_details() -> None:
    result = ma.meta_continuous(DATA, **_columns(), measure="SMD", model="common")
    studies = result.study_results
    options = dict(result.method.options)

    np.testing.assert_allclose(
        studies["pooled_sd"],
        [2.249642828793686, 3.047687648037443, 1.9, 3.811306941989927],
    )
    np.testing.assert_allclose(
        studies["smd_correction_factor"],
        [
            0.986536991051614,
            0.984911942295560,
            0.990093547430093,
            0.988585902542633,
        ],
    )
    np.testing.assert_allclose(
        studies["effect"], studies["cohen_d"] * studies["smd_correction_factor"]
    )
    assert options == {
        "effect_estimator": "hedges_g_exact",
        "standardizer": "pooled_sd",
        "smd_variance": "LS",
    }
    assert result.effect_scale == "identity"
    assert result.display_estimate == result.estimate


def test_md_uses_unpooled_sampling_variance() -> None:
    result = ma.meta_continuous(
        mean_treat=[4.0],
        sd_treat=[2.0],
        n_treat=[20],
        mean_control=[1.5],
        sd_control=[3.0],
        n_control=[30],
        measure="MD",
        model="common",
    )

    assert result.study_results.loc[0, "effect"] == pytest.approx(2.5)
    assert result.study_results.loc[0, "variance"] == pytest.approx(
        2.0**2 / 20 + 3.0**2 / 30
    )
    assert dict(result.method.options) == {"sampling_variance": "unpooled"}


def test_dataframe_defaults_to_index_and_supports_random_effects() -> None:
    data = DATA.drop(columns="study").copy()
    data.index = pd.Index(["A", "B", "C", "D"], name="trial")
    result = ma.meta_continuous(
        data,
        **_columns(),
        measure="SMD",
        model="random",
        tau2_method="PM",
    )

    assert result.study_results["study"].tolist() == ["A", "B", "C", "D"]
    assert result.model == "random"
    assert result.method.tau2_method == "PM"
    assert result.prediction_interval is not None


def test_missing_rows_are_retained_with_specific_reasons() -> None:
    result = ma.meta_continuous(
        mean_treat=[1.0, np.nan, 3.0],
        sd_treat=[1.0, 1.0, 1.0],
        n_treat=[10, 10, 10],
        mean_control=[0.0, 1.0, 2.0],
        sd_control=[1.0, np.nan, 1.0],
        n_control=[10, 10, 10],
        measure="MD",
        model="common",
        missing="drop",
    )

    studies = result.study_results
    assert result.k == 2
    assert studies.loc[1, "exclusion_reason"] == ("missing mean_treat, sd_control")
    assert np.isnan(studies.loc[1, "weight"])
    assert any("Excluded 1 study" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ({"sd_treat": [-1.0]}, "non-negative"),
        ({"n_treat": [1.5]}, "whole-number"),
        ({"n_treat": [1]}, "at least 2"),
        ({"mean_treat": [np.inf]}, "must be finite"),
        ({"mean_treat": ["high"]}, "numeric values"),
    ],
)
def test_invalid_continuous_inputs_raise_domain_errors(
    replacement: dict[str, list[object]], match: str
) -> None:
    inputs: dict[str, list[object]] = {
        "mean_treat": [2.0],
        "sd_treat": [1.0],
        "n_treat": [10],
        "mean_control": [1.0],
        "sd_control": [1.0],
        "n_control": [10],
    }
    inputs.update(replacement)

    with pytest.raises(ma.InvalidStudyDataError, match=match):
        ma.meta_continuous(**inputs, measure="MD", model="common")  # type: ignore[arg-type]


def test_smd_rejects_zero_pooled_sd_and_md_rejects_zero_variance() -> None:
    base = {
        "mean_treat": [2.0],
        "sd_treat": [0.0],
        "n_treat": [10],
        "mean_control": [1.0],
        "sd_control": [0.0],
        "n_control": [10],
        "model": "common",
    }
    with pytest.raises(ma.InvalidStudyDataError, match="positive pooled SD"):
        ma.meta_continuous(**base, measure="SMD")
    with pytest.raises(ma.InvalidStudyDataError, match="strictly positive"):
        ma.meta_continuous(**base, measure="MD")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"measure": "ROM"}, "measure must be"),
        ({"measure": "SMD", "smd_variance": "UB"}, "only 'LS'"),
    ],
)
def test_unsupported_continuous_options(kwargs: dict[str, str], match: str) -> None:
    with pytest.raises(ma.UnsupportedMethodError, match=match):
        ma.meta_continuous(
            mean_treat=[2.0, 3.0],
            sd_treat=[1.0, 1.0],
            n_treat=[10, 10],
            mean_control=[1.0, 2.0],
            sd_control=[1.0, 1.0],
            n_control=[10, 10],
            model="common",
            **kwargs,
        )


def test_continuous_vectors_must_have_equal_lengths() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="equal lengths"):
        ma.meta_continuous(
            mean_treat=[2.0, 3.0],
            sd_treat=[1.0],
            n_treat=[10, 10],
            mean_control=[1.0, 2.0],
            sd_control=[1.0, 1.0],
            n_control=[10, 10],
            model="common",
        )


def test_array_inputs_with_dataframe_must_match_frame_length() -> None:
    data = pd.DataFrame({"unused": [1, 2, 3]})
    with pytest.raises(ma.InvalidStudyDataError, match="one value per DataFrame row"):
        ma.meta_continuous(
            data,
            mean_treat=[2.0, 3.0],
            sd_treat=[1.0, 1.0],
            n_treat=[10, 10],
            mean_control=[1.0, 2.0],
            sd_control=[1.0, 1.0],
            n_control=[10, 10],
            model="common",
        )


def test_missing_policy_errors_and_all_missing_data() -> None:
    base = {
        "mean_treat": [np.nan],
        "sd_treat": [1.0],
        "n_treat": [10],
        "mean_control": [1.0],
        "sd_control": [1.0],
        "n_control": [10],
        "model": "common",
    }
    with pytest.raises(ma.InvalidStudyDataError, match="Missing continuous"):
        ma.meta_continuous(**base)
    with pytest.raises(ma.InvalidStudyDataError, match="No studies remain"):
        ma.meta_continuous(**base, missing="drop")


@pytest.mark.parametrize("measure", ["MD", "SMD"])
def test_swapping_groups_negates_effect_and_reverses_interval(measure: str) -> None:
    forward = ma.meta_continuous(DATA, **_columns(), measure=measure, model="common")
    reverse = ma.meta_continuous(
        mean_treat=DATA["mean_control"],
        sd_treat=DATA["sd_control"],
        n_treat=DATA["n_control"],
        mean_control=DATA["mean_treat"],
        sd_control=DATA["sd_treat"],
        n_control=DATA["n_treat"],
        measure=measure,
        model="common",
    )

    assert reverse.estimate == pytest.approx(-forward.estimate)
    assert reverse.ci == pytest.approx((-forward.ci_high, -forward.ci_low))
    np.testing.assert_allclose(
        reverse.study_results["normalized_weight"],
        forward.study_results["normalized_weight"],
    )


def test_smd_is_invariant_to_common_positive_rescaling() -> None:
    original = ma.meta_continuous(DATA, **_columns(), measure="SMD", model="common")
    scaled = ma.meta_continuous(
        mean_treat=DATA["mean_treat"] * 7.0,
        sd_treat=DATA["sd_treat"] * 7.0,
        n_treat=DATA["n_treat"],
        mean_control=DATA["mean_control"] * 7.0,
        sd_control=DATA["sd_control"] * 7.0,
        n_control=DATA["n_control"],
        measure="SMD",
        model="common",
    )

    assert scaled.estimate == pytest.approx(original.estimate)
    assert scaled.ci == pytest.approx(original.ci)
