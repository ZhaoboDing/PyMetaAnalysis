from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_DATA = pd.read_csv(REFERENCE_DIR / "meta_regression_input.csv")
REFERENCE: dict[str, Any] = json.loads(
    (REFERENCE_DIR / "meta_regression_influence_metafor.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize(
    ("reference_name", "options"),
    [
        ("common", {"model": "common"}),
        ("mixed_dl", {"model": "mixed", "tau2_method": "DL"}),
        ("mixed_pm", {"model": "mixed", "tau2_method": "PM"}),
        ("mixed_reml", {"model": "mixed", "tau2_method": "REML"}),
        (
            "mixed_reml_hartung_knapp",
            {
                "model": "mixed",
                "tau2_method": "REML",
                "inference_method": "hartung_knapp",
            },
        ),
        (
            "mixed_reml_hartung_knapp_adhoc",
            {
                "model": "mixed",
                "tau2_method": "REML",
                "inference_method": "hartung_knapp_adhoc",
            },
        ),
    ],
)
def test_meta_regression_influence_matches_metafor(
    reference_name: str, options: dict[str, str]
) -> None:
    result = ma.meta_regression(
        REFERENCE_DATA,
        effect="effect",
        variance="variance",
        moderators=["mean_age"],
        study="study",
        atol=1e-10,
        max_iter=1000,
        **options,  # type: ignore[arg-type]
    )

    diagnostics = result.influence()
    expected = REFERENCE["models"][reference_name]

    assert REFERENCE["metafor_version"] == "5.0.1"
    assert diagnostics.table["refit_success"].all()
    for column in [
        "deleted_residual",
        "deleted_residual_se",
        "externally_standardized_residual",
        "cook_distance",
    ]:
        np.testing.assert_allclose(
            diagnostics.table[column],
            expected[column],
            rtol=2e-8,
            atol=1e-10,
        )
    actual_dfbetas = (
        diagnostics.dfbetas["dfbetas"].to_numpy().reshape(result.k, result.p)
    )
    np.testing.assert_allclose(
        actual_dfbetas,
        expected["dfbetas"],
        rtol=2e-8,
        atol=1e-10,
    )


def test_meta_regression_influence_dfbetas_use_full_minus_deleted_sign() -> None:
    result = ma.meta_regression(
        effect=[-0.2, 0.1, 0.45, 0.8, 1.05, 1.4],
        variance=[0.04, 0.06, 0.05, 0.08, 0.07, 0.09],
        moderators={"dose": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]},
        model="common",
    )

    diagnostics = result.influence()
    first_refit = diagnostics.results[0]

    assert first_refit is not None
    first_rows = diagnostics.dfbetas.loc[diagnostics.dfbetas["omitted_row_id"] == 0]
    expected_change = (
        result.coefficients["estimate"].to_numpy()
        - first_refit.coefficients["estimate"].to_numpy()
    )
    np.testing.assert_allclose(first_rows["dfbeta"], expected_change)
    np.testing.assert_allclose(
        first_rows["dfbetas"],
        first_rows["dfbeta"] / first_rows["standard_error_reference"],
    )


def test_meta_regression_influence_thresholds_are_explicit_screening_rules() -> None:
    result = ma.meta_regression(
        effect=[0.0, 0.2, 0.4, 0.6, 0.8, 5.0, 1.2, 1.4],
        variance=[0.05] * 8,
        moderators={"dose": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
        model="common",
    )

    diagnostics = result.influence()
    table = diagnostics.table
    dfbetas = diagnostics.dfbetas

    assert diagnostics.studentized_residual_reference == pytest.approx(norm.ppf(0.975))
    assert diagnostics.cook_distance_threshold == pytest.approx(
        chi2.ppf(0.5, df=result.p)
    )
    assert diagnostics.dfbetas_threshold == 1.0
    assert (
        table["studentized_residual_reference"]
        .eq(diagnostics.studentized_residual_reference)
        .all()
    )
    assert (
        table["cook_distance_threshold"].eq(diagnostics.cook_distance_threshold).all()
    )
    assert table["dfbetas_threshold"].eq(diagnostics.dfbetas_threshold).all()
    assert dfbetas["threshold"].eq(diagnostics.dfbetas_threshold).all()

    np.testing.assert_array_equal(
        table["potential_outlier"],
        table["externally_standardized_residual"].abs()
        > diagnostics.studentized_residual_reference,
    )
    np.testing.assert_array_equal(
        table["cook_distance_flag"],
        table["cook_distance"] > diagnostics.cook_distance_threshold,
    )
    grouped_dfbetas = (
        dfbetas.groupby("omitted_row_id", sort=False)["exceeds_threshold"]
        .any()
        .to_numpy()
    )
    np.testing.assert_array_equal(table["dfbetas_flag"], grouped_dfbetas)
    np.testing.assert_array_equal(
        table["potentially_influential"],
        table["cook_distance_flag"] | table["dfbetas_flag"],
    )
    np.testing.assert_array_equal(
        table["flagged"],
        table["potential_outlier"] | table["potentially_influential"],
    )
    assert not diagnostics.flagged.empty
    assert result.study_results["included"].all()


def test_meta_regression_influence_retains_failed_deletions() -> None:
    frame = pd.DataFrame(
        {
            "yi": [-0.2, 0.0, 0.3, 0.5, 0.8, 1.0, 1.2],
            "vi": [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.07],
            "region": ["A", "B", "B", "B", "C", "C", "C"],
        },
        index=[f"Study {index}" for index in range(7)],
    )
    result = ma.meta_regression(
        frame,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["A", "B", "C"]},
        model="common",
    )

    diagnostics = result.influence()

    assert diagnostics.results[0] is None
    assert len(diagnostics.failed) == 1
    failed = diagnostics.failed.iloc[0]
    assert failed["omitted_row_id"] == 0
    assert failed["error_type"] == "InvalidStudyDataError"
    assert "declared levels absent" in failed["error_message"]
    assert np.isnan(failed["externally_standardized_residual"])
    assert np.isnan(failed["cook_distance"])
    assert not failed["flagged"]
    failed_dfbetas = diagnostics.dfbetas.loc[diagnostics.dfbetas["omitted_row_id"] == 0]
    assert len(failed_dfbetas) == result.p
    assert failed_dfbetas["dfbetas"].isna().all()
    assert not failed_dfbetas["exceeds_threshold"].any()
    assert diagnostics.warnings == diagnostics.leave_one_out.warnings


def test_meta_regression_influence_requires_estimable_deleted_models() -> None:
    result = ma.meta_regression(
        effect=[0.0, 0.4, 0.9],
        variance=[0.04, 0.05, 0.06],
        moderators={"dose": [0.0, 1.0, 2.0]},
        model="common",
    )

    with pytest.raises(ma.InsufficientStudiesError, match=r"k >= p \+ 2"):
        result.influence()


def test_meta_regression_influence_preserves_global_ids_after_missing_drop() -> None:
    frame = pd.DataFrame(
        {
            "study": [f"Study {index}" for index in range(7)],
            "yi": [-0.2, np.nan, 0.3, 0.6, 0.9, 1.2, 1.4],
            "vi": [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.07],
            "dose": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        }
    )
    result = ma.meta_regression(
        frame,
        effect="yi",
        variance="vi",
        moderators=["dose"],
        study="study",
        missing="drop",
        model="common",
    )

    diagnostics = result.influence()

    assert diagnostics.table["omitted_row_id"].tolist() == [0, 2, 3, 4, 5, 6]
    assert diagnostics.dfbetas["omitted_row_id"].unique().tolist() == [
        0,
        2,
        3,
        4,
        5,
        6,
    ]
    assert all(
        refit is not None and 1 not in refit.study_results["row_id"].tolist()
        for refit in diagnostics.results
    )


def test_meta_regression_influence_is_stable_under_row_permutation() -> None:
    frame = REFERENCE_DATA.iloc[:8].copy()

    def fit(data: pd.DataFrame) -> ma.MetaRegressionInfluenceResult:
        return ma.meta_regression(
            data,
            effect="effect",
            variance="variance",
            moderators=["mean_age", "dose"],
            study="study",
            model="mixed",
            tau2_method="REML",
        ).influence()

    original = fit(frame)
    permuted = fit(frame.iloc[[4, 1, 7, 0, 6, 3, 5, 2]].reset_index(drop=True))
    original_table = original.table.sort_values("omitted_study").reset_index(drop=True)
    permuted_table = permuted.table.sort_values("omitted_study").reset_index(drop=True)
    for column in [
        "deleted_residual",
        "deleted_residual_se",
        "externally_standardized_residual",
        "cook_distance",
        "max_abs_dfbetas",
    ]:
        np.testing.assert_allclose(
            original_table[column],
            permuted_table[column],
            rtol=2e-10,
            atol=2e-12,
        )
    original_dfbetas = original.dfbetas.sort_values(
        ["omitted_study", "term"]
    ).reset_index(drop=True)
    permuted_dfbetas = permuted.dfbetas.sort_values(
        ["omitted_study", "term"]
    ).reset_index(drop=True)
    np.testing.assert_allclose(
        original_dfbetas["dfbetas"],
        permuted_dfbetas["dfbetas"],
        rtol=2e-10,
        atol=2e-12,
    )


def test_meta_regression_influence_outputs_are_defensive_and_frozen() -> None:
    result = ma.meta_regression(
        effect=[-0.2, 0.1, 0.45, 0.8, 1.05, 1.4],
        variance=[0.04, 0.06, 0.05, 0.08, 0.07, 0.09],
        moderators={"dose": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]},
        model="common",
    )
    diagnostics = result.influence()

    assert isinstance(diagnostics, ma.MetaRegressionInfluenceResult)
    assert len(diagnostics) == result.k
    pd.testing.assert_frame_equal(diagnostics.summary(), diagnostics.table)
    pd.testing.assert_frame_equal(diagnostics.to_dataframe(), diagnostics.table)
    table = diagnostics.table
    table.loc[0, "cook_distance"] = 999.0
    assert diagnostics.table.loc[0, "cook_distance"] != 999.0
    dfbetas = diagnostics.dfbetas
    dfbetas.loc[0, "dfbetas"] = 999.0
    assert diagnostics.dfbetas.loc[0, "dfbetas"] != 999.0
    with pytest.raises(FrozenInstanceError):
        diagnostics.dfbetas_threshold = 2.0  # type: ignore[misc]
