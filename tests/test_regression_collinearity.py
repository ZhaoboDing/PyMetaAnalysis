from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_DATA = pd.read_csv(REFERENCE_DIR / "meta_regression_input.csv")
REFERENCE: dict[str, Any] = json.loads(
    (REFERENCE_DIR / "meta_regression_collinearity_metafor.json").read_text(
        encoding="utf-8"
    )
)


def _reference_fit(**options: str) -> ma.MetaRegressionResult:
    return ma.meta_regression(
        REFERENCE_DATA,
        effect="effect",
        variance="variance",
        moderators=["mean_age", "dose", "region"],
        categorical={"region": REFERENCE["categorical_levels"]["region"]},
        study="study",
        atol=1e-10,
        max_iter=1000,
        **options,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("reference_name", "options"),
    [
        ("common", {"model": "common"}),
        ("mixed_reml", {"model": "mixed", "tau2_method": "REML"}),
    ],
)
def test_meta_regression_vif_and_gvif_match_metafor(
    reference_name: str, options: dict[str, str]
) -> None:
    diagnostics = _reference_fit(**options).collinearity()
    expected = REFERENCE["models"][reference_name]

    assert REFERENCE["metafor_version"] == "5.0.1"
    assert diagnostics.term_vif["term"].tolist() == [
        "mean_age",
        "dose",
        "region[South]",
        "region[East]",
    ]
    assert (
        diagnostics.moderator_gvif["moderator"].tolist() == REFERENCE["moderator_names"]
    )
    np.testing.assert_allclose(
        diagnostics.term_vif["vif"],
        expected["term_vif"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        diagnostics.term_vif["sif"],
        expected["term_sif"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        diagnostics.moderator_gvif["gvif"],
        expected["moderator_gvif"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        diagnostics.moderator_gvif["gsif"],
        expected["moderator_gsif"],
        rtol=2e-10,
        atol=2e-12,
    )


def test_condition_diagnostics_match_manual_weighted_svd() -> None:
    result = _reference_fit(model="mixed", tau2_method="REML")
    diagnostics = result.collinearity()
    studies = result.study_results
    included = studies["included"].to_numpy(dtype=np.bool_)
    variance = studies.loc[included, "variance"].to_numpy(dtype=np.float64)
    design = result.design_matrix.to_numpy(dtype=np.float64)
    weighted_design = np.sqrt(1.0 / (variance + result.tau2))[:, np.newaxis] * design
    scaled_design = weighted_design / np.linalg.norm(weighted_design, axis=0)
    _, singular_values, right_vectors_transposed = np.linalg.svd(
        scaled_design, full_matrices=False
    )
    eigenvalues = singular_values**2
    expected_indices = singular_values[0] / singular_values
    variance_components = right_vectors_transposed.T**2 / eigenvalues[np.newaxis, :]
    expected_proportions = variance_components / variance_components.sum(
        axis=1, keepdims=True
    )
    actual_proportions = (
        diagnostics.variance_proportions.pivot(
            index="term",
            columns="dimension",
            values="variance_proportion",
        )
        .loc[list(result.design_info.term_names)]
        .to_numpy()
    )

    np.testing.assert_allclose(
        diagnostics.condition_indices["singular_value"], singular_values
    )
    np.testing.assert_allclose(
        diagnostics.condition_indices["condition_index"], expected_indices
    )
    np.testing.assert_allclose(actual_proportions, expected_proportions)
    np.testing.assert_allclose(actual_proportions.sum(axis=1), 1.0)
    assert diagnostics.weighted_scaled_condition_number == pytest.approx(
        expected_indices[-1]
    )
    assert diagnostics.raw_condition_number == result.diagnostics.condition_number


def test_single_moderator_and_no_intercept_have_unit_inflation() -> None:
    result = ma.meta_regression(
        effect=[0.1, 0.3, 0.4, 0.7, 0.9],
        variance=[0.04, 0.05, 0.06, 0.05, 0.07],
        moderators={"dose": [1.0, 2.0, 3.0, 4.0, 5.0]},
        model="common",
        intercept=False,
    )

    diagnostics = result.collinearity()

    assert diagnostics.term_vif.loc[0, "vif"] == pytest.approx(1.0)
    assert diagnostics.term_vif.loc[0, "sif"] == pytest.approx(1.0)
    assert diagnostics.moderator_gvif.loc[0, "gvif"] == pytest.approx(1.0)
    assert diagnostics.moderator_gvif.loc[0, "gsif"] == pytest.approx(1.0)
    assert diagnostics.weighted_scaled_condition_number == pytest.approx(1.0)
    assert diagnostics.warnings == ()


def test_categorical_terms_are_grouped_by_original_moderator() -> None:
    diagnostics = _reference_fit(model="common").collinearity()
    region = diagnostics.moderator_gvif.set_index("moderator").loc["region"]

    assert region["kind"] == "categorical"
    assert region["terms"] == ("region[South]", "region[East]")
    assert region["term_count"] == 2
    assert diagnostics.term_vif.loc[
        diagnostics.term_vif["moderator"] == "region", "term"
    ].tolist() == ["region[South]", "region[East]"]


def test_diagnostics_are_invariant_to_positive_rescaling_and_row_order() -> None:
    original = REFERENCE_DATA.copy()
    rescaled = original.copy()
    rescaled["mean_age"] *= 100.0
    rescaled["dose"] *= 0.01
    permuted = original.iloc[
        [7, 2, 12, 0, 14, 6, 4, 10, 1, 9, 5, 13, 3, 11, 8]
    ].reset_index(drop=True)

    def fit(data: pd.DataFrame) -> ma.MetaRegressionCollinearityResult:
        return ma.meta_regression(
            data,
            effect="effect",
            variance="variance",
            moderators=["mean_age", "dose", "region"],
            categorical={"region": ["North", "South", "East"]},
            study="study",
            model="mixed",
            tau2_method="REML",
        ).collinearity()

    baseline = fit(original)
    for alternative in (fit(rescaled), fit(permuted)):
        np.testing.assert_allclose(
            alternative.term_vif["vif"], baseline.term_vif["vif"], rtol=2e-10
        )
        np.testing.assert_allclose(
            alternative.moderator_gvif["gvif"],
            baseline.moderator_gvif["gvif"],
            rtol=2e-10,
        )
        np.testing.assert_allclose(
            alternative.condition_indices["condition_index"],
            baseline.condition_indices["condition_index"],
            rtol=2e-10,
        )
    assert fit(rescaled).raw_condition_number != pytest.approx(
        baseline.raw_condition_number
    )


def test_missing_drop_uses_only_the_fitted_rows() -> None:
    data = REFERENCE_DATA.iloc[:10].copy()
    data.loc[3, "effect"] = np.nan
    dropped = ma.meta_regression(
        data,
        effect="effect",
        variance="variance",
        moderators=["mean_age", "dose"],
        study="study",
        model="common",
        missing="drop",
    ).collinearity()
    filtered = ma.meta_regression(
        data.dropna(subset=["effect"]).reset_index(drop=True),
        effect="effect",
        variance="variance",
        moderators=["mean_age", "dose"],
        study="study",
        model="common",
    ).collinearity()

    np.testing.assert_allclose(dropped.term_vif["vif"], filtered.term_vif["vif"])
    np.testing.assert_allclose(
        dropped.condition_indices["condition_index"],
        filtered.condition_indices["condition_index"],
    )


def test_near_collinearity_is_flagged_without_changing_the_model() -> None:
    x = np.arange(12, dtype=np.float64)
    z = x + np.asarray(
        [
            -0.01,
            0.01,
            -0.015,
            0.012,
            -0.008,
            0.018,
            -0.012,
            0.01,
            -0.018,
            0.014,
            -0.009,
            0.011,
        ]
    )
    effect = (
        0.2
        + 0.3 * x
        - 0.2 * z
        + np.asarray(
            [0.0, 0.02, -0.01, 0.01, -0.02, 0.01, 0.0, -0.01, 0.02, -0.02, 0.01, 0.0]
        )
    )
    result = ma.meta_regression(
        effect=effect,
        variance=np.linspace(0.04, 0.08, 12),
        moderators={"x": x, "z": z},
        model="common",
    )

    diagnostics = result.collinearity()

    assert diagnostics.weighted_scaled_condition_number > 30.0
    assert diagnostics.term_vif["vif"].gt(1000.0).all()
    assert not diagnostics.concerning_dimensions.empty
    assert len(diagnostics.warnings) == 2
    assert result.design_info.term_names == ("intercept", "x", "z")
    assert result.p == 3


def test_collinearity_outputs_are_defensive_and_frozen() -> None:
    diagnostics = _reference_fit(model="common").collinearity()

    assert isinstance(diagnostics, ma.MetaRegressionCollinearityResult)
    term_vif = diagnostics.term_vif
    term_vif.loc[0, "vif"] = 999.0
    assert diagnostics.term_vif.loc[0, "vif"] != 999.0
    moderator_gvif = diagnostics.moderator_gvif
    moderator_gvif.loc[0, "gvif"] = 999.0
    assert diagnostics.moderator_gvif.loc[0, "gvif"] != 999.0
    condition_indices = diagnostics.condition_indices
    condition_indices.loc[0, "condition_index"] = 999.0
    assert diagnostics.condition_indices.loc[0, "condition_index"] != 999.0
    variance_proportions = diagnostics.variance_proportions
    variance_proportions.loc[0, "variance_proportion"] = 999.0
    assert diagnostics.variance_proportions.loc[0, "variance_proportion"] != 999.0
    with pytest.raises(FrozenInstanceError):
        diagnostics.condition_index_reference = 10.0  # type: ignore[misc]
