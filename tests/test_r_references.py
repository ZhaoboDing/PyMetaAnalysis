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
WORKFLOW_DATA = pd.read_csv(REFERENCE_DIR / "workflow_input.csv")
META_REGRESSION_DATA = pd.read_csv(REFERENCE_DIR / "meta_regression_input.csv")


def _load_reference(filename: str) -> dict[str, Any]:
    return json.loads((REFERENCE_DIR / filename).read_text(encoding="utf-8"))


GENERIC = _load_reference("generic_metafor.json")
BINARY = _load_reference("binary_metafor.json")
CONTINUOUS = _load_reference("continuous_metafor.json")
WORKFLOW = _load_reference("workflow_metafor.json")
META_REGRESSION = _load_reference("meta_regression_metafor.json")

CLOSED_RTOL = 5e-13
CLOSED_ATOL = 5e-15
ITERATIVE_RTOL = 2e-10
ITERATIVE_ATOL = 2e-11
ITERATIVE_DERIVED_RTOL = 2e-9
ITERATIVE_DERIVED_ATOL = 1e-10


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
    fixtures = [GENERIC, BINARY, CONTINUOUS, WORKFLOW, META_REGRESSION]

    assert {fixture["generated_by"] for fixture in fixtures} == {"R metafor"}
    assert len({fixture["r_version"] for fixture in fixtures}) == 1
    assert len({fixture["metafor_version"] for fixture in fixtures}) == 1
    assert len({fixture["jsonlite_version"] for fixture in fixtures}) == 1
    assert GENERIC["iterative_control"] == {
        "tolerance": 1e-10,
        "max_iterations": 1000,
    }
    assert BINARY["iterative_control"] == GENERIC["iterative_control"]
    assert WORKFLOW["iterative_control"] == GENERIC["iterative_control"]
    assert META_REGRESSION["iterative_control"] == GENERIC["iterative_control"]


def _assert_regression_fit(
    result: ma.MetaRegressionResult,
    expected: dict[str, Any],
    *,
    iterative: bool = False,
    compare_heterogeneity: bool = True,
) -> None:
    rtol = ITERATIVE_RTOL if iterative else CLOSED_RTOL
    atol = ITERATIVE_ATOL if iterative else CLOSED_ATOL
    derived_rtol = ITERATIVE_DERIVED_RTOL if iterative else CLOSED_RTOL
    derived_atol = ITERATIVE_DERIVED_ATOL if iterative else CLOSED_ATOL
    coefficients = result.coefficients
    expected_coefficients = expected["coefficients"]

    assert coefficients["term"].tolist() == expected["term_names"]
    for column in [
        "estimate",
        "standard_error",
        "statistic",
        "pvalue",
        "ci_low",
        "ci_high",
    ]:
        np.testing.assert_allclose(
            coefficients[column],
            expected_coefficients[column],
            rtol=rtol,
            atol=atol,
        )
    np.testing.assert_allclose(
        result.coefficient_covariance,
        expected_coefficients["covariance"],
        rtol=rtol,
        atol=atol,
    )

    assert result.tau2 == pytest.approx(expected["tau2"], rel=rtol, abs=atol)
    if result.model == "common":
        assert expected["tau2_null"] is None
        assert expected["pseudo_r2"] is None
        assert expected["pseudo_r2_raw"] is None
        assert result.tau2_null is None
        assert result.pseudo_r2 is None
        assert result.pseudo_r2_raw is None
    else:
        assert result.tau2_null == pytest.approx(
            expected["tau2_null"], rel=rtol, abs=atol
        )
        assert result.pseudo_r2 == pytest.approx(
            expected["pseudo_r2"], rel=derived_rtol, abs=derived_atol
        )
        assert result.pseudo_r2_raw == pytest.approx(
            expected["pseudo_r2_raw"], rel=derived_rtol, abs=derived_atol
        )

    if compare_heterogeneity:
        heterogeneity = expected["residual_heterogeneity"]
        np.testing.assert_allclose(
            [
                result.heterogeneity.q,
                result.heterogeneity.pvalue,
            ],
            [
                heterogeneity["q"],
                heterogeneity["pvalue"],
            ],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            [result.heterogeneity.i2, result.heterogeneity.h2],
            [
                heterogeneity["i2"],
                heterogeneity["h2"],
            ],
            rtol=derived_rtol,
            atol=derived_atol,
        )
        assert result.heterogeneity.df == heterogeneity["df"]

    moderator_test = expected["global_moderator_test"]
    np.testing.assert_allclose(
        [result.global_test.statistic, result.global_test.pvalue],
        [moderator_test["statistic"], moderator_test["pvalue"]],
        rtol=rtol,
        atol=atol,
    )
    assert result.global_test.df_num == moderator_test["df_num"]
    assert result.global_test.df_denom == moderator_test["df_denom"]
    assert result.diagnostics.residual_scale == pytest.approx(
        expected["residual_scale"], rel=rtol, abs=atol
    )

    studies = result.study_results
    for column, reference_key in [
        ("normalized_precision_weight", "normalized_weights"),
        ("fitted_value", "fitted_values"),
        ("residual", "residuals"),
        ("leverage", "leverage"),
    ]:
        np.testing.assert_allclose(
            studies[column],
            expected[reference_key],
            rtol=rtol,
            atol=atol,
        )
    assert studies["normalized_precision_weight"].sum() == pytest.approx(1.0, abs=2e-15)


def _assert_summary_values(
    result: ma.MetaAnalysisResult,
    expected: dict[str, Any],
    *,
    iterative: bool = False,
) -> None:
    rtol = ITERATIVE_RTOL if iterative else CLOSED_RTOL
    atol = ITERATIVE_ATOL if iterative else CLOSED_ATOL
    np.testing.assert_allclose(
        [
            result.estimate,
            result.standard_error,
            result.ci_low,
            result.ci_high,
            result.tau2,
        ],
        [
            expected["estimate"],
            expected["standard_error"],
            *expected["ci"],
            expected["tau2"],
        ],
        rtol=rtol,
        atol=atol,
    )


def _assert_workflow_table(
    table: pd.DataFrame,
    expected: dict[str, Any],
    *,
    expected_slice: slice | None = None,
    iterative: bool = False,
) -> None:
    rtol = ITERATIVE_RTOL if iterative else CLOSED_RTOL
    atol = ITERATIVE_ATOL if iterative else CLOSED_ATOL
    selected = slice(None) if expected_slice is None else expected_slice
    for column in [
        "estimate",
        "standard_error",
        "ci_low",
        "ci_high",
        "tau2",
        "q",
    ]:
        np.testing.assert_allclose(
            table[column],
            np.asarray(expected[column], dtype=np.float64)[selected],
            rtol=rtol,
            atol=atol,
        )
    valid_q_pvalue = np.ones(len(table), dtype=bool)
    if "q_df" in table:
        valid_q_pvalue = table["q_df"].to_numpy() > 0
    np.testing.assert_allclose(
        table.loc[valid_q_pvalue, "q_pvalue"],
        np.asarray(expected["q_pvalue"], dtype=np.float64)[selected][valid_q_pvalue],
        rtol=rtol,
        atol=atol,
    )


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

    mh_expected = BINARY["clean"][measure]["mantel_haenszel"]
    _assert_fit(result, mh_expected)
    expected = mh_expected["heterogeneity"]
    np.testing.assert_allclose(
        [result.q, result.q_pvalue, result.i2, result.h2],
        [expected["q"], expected["pvalue"], expected["i2"], expected["h2"]],
        rtol=CLOSED_RTOL,
        atol=CLOSED_ATOL,
    )
    assert result.q_df == expected["df"]
    assert not result.study_results["mh_continuity_corrected"].any()


def test_sparse_risk_difference_variances_match_metafor_boundary_tables() -> None:
    expected = BINARY["sparse"]["RD"]
    result = ma.meta_binary(
        SPARSE_BINARY_DATA,
        **_binary_columns(),
        study="study",
        measure="RD",
        method="IV",
        model="common",
    )
    studies = result.study_results

    np.testing.assert_allclose(
        studies["variance"], expected["variance"], rtol=CLOSED_RTOL, atol=CLOSED_ATOL
    )
    # Metafor corrects the displayed effect for the single-zero row, while this
    # package deliberately retains the raw RD. The remaining rows agree.
    assert studies.loc[0, "effect"] == pytest.approx(-0.08)
    np.testing.assert_allclose(
        studies.loc[1:, "effect"],
        expected["metafor_effect"][1:],
        rtol=CLOSED_RTOL,
        atol=CLOSED_ATOL,
    )
    assert studies["rd_zero_variance"].tolist() == [False, False, True, True, False]


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


def test_common_effect_subgroups_match_metafor_moderator_model() -> None:
    expected = WORKFLOW["subgroup_common"]
    result = ma.meta_analysis(
        WORKFLOW_DATA,
        effect="effect",
        variance="variance",
        study="study",
        subgroup="subgroup",
        model="common",
    )

    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    _assert_summary_values(result.overall, expected["overall"])
    assert list(result.groups) == list(expected["groups"])
    for name, group in result.groups.items():
        _assert_summary_values(group, expected["groups"][name])
    np.testing.assert_allclose(
        [result.q_between, result.q_between_pvalue, result.i2_between],
        [
            expected["q_between"],
            expected["q_between_pvalue"],
            expected["i2_between"],
        ],
        rtol=CLOSED_RTOL,
        atol=CLOSED_ATOL,
    )
    assert result.q_between_df == expected["q_between_df"]


@pytest.mark.parametrize(
    ("model", "reference_key", "iterative"),
    [("common", "common", False), ("random", "random_reml", True)],
)
def test_leave_one_out_workflow_matches_metafor(
    model: str,
    reference_key: str,
    iterative: bool,
) -> None:
    expected = WORKFLOW["leave_one_out"][reference_key]
    result = ma.meta_analysis(
        WORKFLOW_DATA,
        effect="effect",
        variance="variance",
        study="study",
        model=model,
        tau2_method="REML",
    )
    table = result.leave_one_out().to_dataframe()

    assert table["omitted_study"].tolist() == expected["study"]
    _assert_workflow_table(table, expected, iterative=iterative)
    np.testing.assert_allclose(
        table[["i2", "h2"]],
        np.column_stack([expected["i2"], expected["h2"]]),
        rtol=ITERATIVE_RTOL if iterative else CLOSED_RTOL,
        atol=ITERATIVE_ATOL if iterative else CLOSED_ATOL,
    )


@pytest.mark.parametrize(
    ("model", "reference_key", "start", "iterative"),
    [
        ("common", "common", 0, False),
        ("random", "random_reml", 1, True),
    ],
)
def test_cumulative_workflow_matches_metafor(
    model: str,
    reference_key: str,
    start: int,
    iterative: bool,
) -> None:
    expected = WORKFLOW["cumulative"][reference_key]
    result = ma.meta_analysis(
        WORKFLOW_DATA,
        effect="effect",
        variance="variance",
        study="study",
        model=model,
        tau2_method="REML",
    )
    table = result.cumulative(order="year").to_dataframe()
    selected = slice(start, None)

    assert table["k"].tolist() == expected["k"][selected]
    assert table["order_value"].tolist() == expected["year"][selected]
    added_studies = [study for studies in table["added_studies"] for study in studies]
    assert added_studies == expected["study"]
    _assert_workflow_table(
        table,
        expected,
        expected_slice=selected,
        iterative=iterative,
    )
    valid = table["q_df"].to_numpy() > 0
    expected_i2 = np.asarray(expected["i2"], dtype=np.float64)[selected][valid]
    expected_h2 = np.asarray(expected["h2"], dtype=np.float64)[selected][valid]
    np.testing.assert_allclose(
        table.loc[valid, ["i2", "h2"]],
        np.column_stack([expected_i2, expected_h2]),
        rtol=ITERATIVE_RTOL if iterative else CLOSED_RTOL,
        atol=ITERATIVE_ATOL if iterative else CLOSED_ATOL,
    )


def test_common_numeric_meta_regression_matches_metafor() -> None:
    result = ma.meta_regression(
        META_REGRESSION_DATA,
        effect="effect",
        variance="variance",
        study="study",
        moderators=["mean_age"],
        model="common",
    )

    _assert_regression_fit(result, META_REGRESSION["common_numeric"])


@pytest.mark.parametrize("tau2_method", ["DL", "PM", "REML"])
def test_mixed_numeric_meta_regression_matches_metafor(
    tau2_method: str,
) -> None:
    result = ma.meta_regression(
        META_REGRESSION_DATA,
        effect="effect",
        variance="variance",
        study="study",
        moderators=["mean_age"],
        model="mixed",
        tau2_method=tau2_method,
    )

    _assert_regression_fit(
        result,
        META_REGRESSION["mixed_numeric"][tau2_method],
        iterative=tau2_method in {"PM", "REML"},
    )


@pytest.mark.parametrize(
    "inference_method",
    ["hartung_knapp", "hartung_knapp_adhoc"],
)
def test_meta_regression_hartung_knapp_variants_match_metafor(
    inference_method: str,
) -> None:
    result = ma.meta_regression(
        META_REGRESSION_DATA,
        effect="effect",
        variance="variance",
        study="study",
        moderators=["mean_age"],
        model="mixed",
        tau2_method="REML",
        inference_method=inference_method,
    )

    _assert_regression_fit(
        result,
        META_REGRESSION["mixed_numeric_inference"][inference_method],
        iterative=True,
    )


def test_hartung_knapp_meta_regression_predictions_match_metafor() -> None:
    expected = META_REGRESSION["mixed_numeric_inference"]["hartung_knapp"]
    result = ma.meta_regression(
        META_REGRESSION_DATA,
        effect="effect",
        variance="variance",
        study="study",
        moderators=["mean_age"],
        model="mixed",
        tau2_method="REML",
        inference_method="hartung_knapp",
    )
    predictions = result.predict(
        pd.DataFrame({"mean_age": expected["prediction_values"]})
    )

    for column in predictions:
        np.testing.assert_allclose(
            predictions[column],
            expected["predictions"][column],
            rtol=ITERATIVE_RTOL,
            atol=ITERATIVE_ATOL,
        )


def test_no_intercept_meta_regression_fit_matches_metafor() -> None:
    result = ma.meta_regression(
        META_REGRESSION_DATA,
        effect="effect",
        variance="variance",
        study="study",
        moderators=["mean_age"],
        model="common",
        intercept=False,
    )

    _assert_regression_fit(
        result,
        META_REGRESSION["common_no_intercept"],
        compare_heterogeneity=False,
    )


def test_categorical_meta_regression_matches_metafor() -> None:
    levels = META_REGRESSION["categorical_levels"]["region"]
    result = ma.meta_regression(
        META_REGRESSION_DATA,
        effect="effect",
        variance="variance",
        study="study",
        moderators=["region"],
        categorical={"region": levels},
        model="common",
    )

    _assert_regression_fit(result, META_REGRESSION["common_categorical"])


def test_multivariable_meta_regression_and_predictions_match_metafor() -> None:
    expected = META_REGRESSION["mixed_multivariable_reml"]
    levels = META_REGRESSION["categorical_levels"]["region"]
    result = ma.meta_regression(
        META_REGRESSION_DATA,
        effect="effect",
        variance="variance",
        study="study",
        moderators=["mean_age", "dose", "region"],
        categorical={"region": levels},
        model="mixed",
        tau2_method="REML",
    )

    _assert_regression_fit(result, expected, iterative=True)

    region_test = result.test_moderator("region")
    np.testing.assert_allclose(
        [region_test.statistic, region_test.pvalue],
        [expected["region_test"]["statistic"], expected["region_test"]["pvalue"]],
        rtol=ITERATIVE_RTOL,
        atol=ITERATIVE_ATOL,
    )
    assert region_test.df_num == expected["region_test"]["df_num"]

    predictions = result.predict(pd.DataFrame(expected["prediction_rows"]))
    for column in predictions:
        np.testing.assert_allclose(
            predictions[column],
            expected["predictions"][column],
            rtol=ITERATIVE_DERIVED_RTOL,
            atol=ITERATIVE_DERIVED_ATOL,
        )
