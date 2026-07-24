from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma


def _numeric_inputs() -> dict[str, np.ndarray]:
    return {
        "effect": np.asarray([-0.2, 0.1, 0.45, 0.8, 1.05, 1.4]),
        "variance": np.asarray([0.04, 0.06, 0.05, 0.08, 0.07, 0.09]),
        "dose": np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
    }


def test_meta_regression_leave_one_out_matches_direct_common_refits() -> None:
    inputs = _numeric_inputs()
    result = ma.meta_regression(
        effect=inputs["effect"],
        variance=inputs["variance"],
        moderators={"dose": inputs["dose"]},
        study=[f"Study {index}" for index in range(6)],
        model="common",
    )

    diagnostics = result.leave_one_out()

    assert isinstance(diagnostics, ma.MetaRegressionLeaveOneOutResult)
    assert len(diagnostics) == 6
    assert diagnostics.table["omitted_row_id"].tolist() == list(range(6))
    assert diagnostics.table["refit_success"].all()
    assert diagnostics.failed.empty
    assert diagnostics.warnings == ()
    for omitted, refit in enumerate(diagnostics.results):
        assert refit is not None
        keep = np.arange(6) != omitted
        direct = ma.meta_regression(
            effect=inputs["effect"][keep],
            variance=inputs["variance"][keep],
            moderators={"dose": inputs["dose"][keep]},
            study=np.asarray([f"Study {index}" for index in range(6)])[keep],
            model="common",
        )
        np.testing.assert_allclose(
            refit.coefficients["estimate"], direct.coefficients["estimate"]
        )
        assert refit.heterogeneity.q == pytest.approx(direct.heterogeneity.q)
        assert refit.study_results["row_id"].tolist() == np.flatnonzero(keep).tolist()
        assert refit.provenance.included_rows == tuple(np.flatnonzero(keep))

    coefficients = diagnostics.coefficients
    assert len(coefficients) == 12
    first_deleted = coefficients.loc[coefficients["omitted_row_id"] == 0]
    expected_change = (
        diagnostics.results[0].coefficients["estimate"].to_numpy()
        - result.coefficients["estimate"].to_numpy()
    )
    np.testing.assert_allclose(first_deleted["estimate_change"], expected_change)


def test_meta_regression_leave_one_out_preserves_mixed_model_configuration() -> None:
    inputs = _numeric_inputs()
    result = ma.meta_regression(
        effect=inputs["effect"],
        variance=inputs["variance"],
        moderators={"dose": inputs["dose"]},
        model="mixed",
        tau2_method="PM",
        inference_method="hartung_knapp_adhoc",
        confidence_level=0.9,
        prediction_interval_method="riley",
        atol=1e-8,
        max_iter=321,
    )

    diagnostics = result.leave_one_out()

    for refit in diagnostics.results:
        assert refit is not None
        assert refit.model == "mixed"
        assert refit.method.tau2_method == "PM"
        assert refit.method.inference_method == "hartung_knapp_adhoc"
        assert refit.method.confidence_level == pytest.approx(0.9)
        assert refit.method.prediction_interval_method == "riley"
        assert refit.method.atol == pytest.approx(1e-8)
        assert refit.method.max_iter == 321
    assert diagnostics.table["global_statistic_name"].eq("F").all()


def test_meta_regression_leave_one_out_records_unidentifiable_deletion() -> None:
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

    diagnostics = result.leave_one_out()

    assert diagnostics.results[0] is None
    assert all(refit is not None for refit in diagnostics.results[1:])
    assert diagnostics.table["refit_success"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    failed = diagnostics.failed.iloc[0]
    assert failed["omitted_row_id"] == 0
    assert failed["error_type"] == "InvalidStudyDataError"
    assert "declared levels absent" in failed["error_message"]
    assert len(diagnostics.warnings) == 1
    failed_coefficients = diagnostics.coefficients.loc[
        diagnostics.coefficients["omitted_row_id"] == 0
    ]
    assert len(failed_coefficients) == result.p
    assert failed_coefficients["estimate"].isna().all()
    assert failed_coefficients["estimate_change"].isna().all()


def test_meta_regression_leave_one_out_keeps_global_ids_after_missing_drop() -> None:
    frame = pd.DataFrame(
        {
            "yi": [-0.2, np.nan, 0.3, 0.6, 0.9, 1.2, 1.4],
            "vi": [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.07],
            "dose": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        },
        index=[f"Study {index}" for index in range(7)],
    )
    result = ma.meta_regression(
        frame,
        effect="yi",
        variance="vi",
        moderators=["dose"],
        missing="drop",
        model="common",
    )

    diagnostics = result.leave_one_out()

    assert diagnostics.table["omitted_row_id"].tolist() == [0, 2, 3, 4, 5, 6]
    for refit in diagnostics.results:
        assert refit is not None
        assert refit.method.missing == "drop"
        assert 1 not in refit.study_results["row_id"].tolist()
        assert refit.source_data is not None
        assert len(refit.source_data) == 5


def test_meta_regression_leave_one_out_supports_no_intercept_multiple_moderators() -> (
    None
):
    result = ma.meta_regression(
        effect=[-0.2, 0.1, 0.4, 0.8, 1.1, 1.5],
        variance=[0.04, 0.06, 0.05, 0.08, 0.07, 0.09],
        moderators={
            "dose": [0.5, 1.0, 1.4, 2.0, 2.5, 3.0],
            "duration": [1.0, 0.5, 1.5, 1.0, 2.0, 1.5],
        },
        intercept=False,
        model="common",
    )

    diagnostics = result.leave_one_out()

    assert diagnostics.table["refit_success"].all()
    assert diagnostics.coefficients["term"].unique().tolist() == [
        "dose",
        "duration",
    ]
    assert all(
        refit is not None and not refit.method.intercept
        for refit in diagnostics.results
    )


def test_meta_regression_leave_one_out_is_stable_under_row_permutation() -> None:
    frame = pd.DataFrame(
        {
            "study": [f"Study {index}" for index in range(6)],
            "yi": [-0.2, 0.1, 0.45, 0.8, 1.05, 1.4],
            "vi": [0.04, 0.06, 0.05, 0.08, 0.07, 0.09],
            "dose": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        }
    )

    def fit(data: pd.DataFrame) -> ma.MetaRegressionLeaveOneOutResult:
        return ma.meta_regression(
            data,
            effect="yi",
            variance="vi",
            moderators=["dose"],
            study="study",
            model="common",
        ).leave_one_out()

    original = fit(frame)
    permuted = fit(frame.iloc[[4, 1, 5, 0, 3, 2]].reset_index(drop=True))
    original_table = original.table.sort_values("omitted_study").reset_index(drop=True)
    permuted_table = permuted.table.sort_values("omitted_study").reset_index(drop=True)
    for column in [
        "tau2",
        "residual_q",
        "residual_q_pvalue",
        "global_statistic",
        "global_pvalue",
        "condition_number",
    ]:
        np.testing.assert_allclose(original_table[column], permuted_table[column])
    original_coefficients = original.coefficients.sort_values(
        ["omitted_study", "term"]
    ).reset_index(drop=True)
    permuted_coefficients = permuted.coefficients.sort_values(
        ["omitted_study", "term"]
    ).reset_index(drop=True)
    np.testing.assert_allclose(
        original_coefficients["estimate"], permuted_coefficients["estimate"]
    )
    np.testing.assert_allclose(
        original_coefficients["estimate_change"],
        permuted_coefficients["estimate_change"],
    )


def test_meta_regression_leave_one_out_requires_one_extra_residual_degree() -> None:
    result = ma.meta_regression(
        effect=[0.0, 0.4, 0.9],
        variance=[0.04, 0.05, 0.06],
        moderators={"dose": [0.0, 1.0, 2.0]},
        model="common",
    )

    with pytest.raises(ma.InsufficientStudiesError, match=r"k >= p \+ 2"):
        result.leave_one_out()


def test_meta_regression_leave_one_out_outputs_are_defensive_and_frozen() -> None:
    inputs = _numeric_inputs()
    result = ma.meta_regression(
        effect=inputs["effect"],
        variance=inputs["variance"],
        moderators={"dose": inputs["dose"]},
        model="common",
    )
    diagnostics = result.leave_one_out()

    table = diagnostics.table
    pd.testing.assert_frame_equal(diagnostics.summary(), diagnostics.table)
    pd.testing.assert_frame_equal(diagnostics.to_dataframe(), diagnostics.table)
    table.loc[0, "tau2"] = 999.0
    assert diagnostics.table.loc[0, "tau2"] != 999.0
    coefficients = diagnostics.coefficients
    coefficients.loc[0, "estimate"] = 999.0
    assert diagnostics.coefficients.loc[0, "estimate"] != 999.0
    with pytest.raises(FrozenInstanceError):
        diagnostics.warnings = ("changed",)  # type: ignore[misc]
