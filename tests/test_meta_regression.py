from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

import meta_analyze as ma
from meta_analyze.estimators import fit_meta_regression


def _regression_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study": [f"Study {index}" for index in range(1, 9)],
            "yi": [-0.30, -0.05, 0.28, 0.55, 0.82, 1.18, 1.35, 1.72],
            "vi": [0.04, 0.06, 0.05, 0.08, 0.04, 0.07, 0.06, 0.09],
            "dose": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "region": [
                "North",
                "South",
                "East",
                "North",
                "South",
                "East",
                "North",
                "South",
            ],
        }
    )


def test_common_meta_regression_matches_hand_calculated_wls() -> None:
    effect = np.asarray([0.2, 0.4, 0.8, 1.1, 1.4], dtype=np.float64)
    variance = np.asarray([0.04, 0.09, 0.05, 0.08, 0.12], dtype=np.float64)
    moderator = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    design = np.column_stack([np.ones(len(effect)), moderator])
    weights = 1.0 / variance
    gram = design.T @ (weights[:, np.newaxis] * design)
    expected_covariance = np.linalg.inv(gram)
    expected_coefficients = expected_covariance @ design.T @ (weights * effect)
    expected_fitted = design @ expected_coefficients
    expected_qe = float(np.dot(weights, (effect - expected_fitted) ** 2))

    result = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"dose": moderator},
        model="common",
    )

    np.testing.assert_allclose(
        result.coefficients["estimate"], expected_coefficients, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result.coefficient_covariance, expected_covariance, rtol=1e-12, atol=1e-12
    )
    assert result.heterogeneity.q == pytest.approx(expected_qe)
    assert result.heterogeneity.df == 3
    assert result.heterogeneity.pvalue == pytest.approx(chi2.sf(expected_qe, 3))
    assert result.study_results["normalized_precision_weight"].sum() == pytest.approx(
        1.0
    )
    assert result.tau2 == 0.0
    assert result.method.tau2_method is None


@pytest.mark.parametrize("tau2_method", ["DL", "PM", "REML"])
@pytest.mark.parametrize(
    "inference_method", ["normal", "hartung_knapp", "hartung_knapp_adhoc"]
)
def test_intercept_only_numerical_core_matches_meta_analysis(
    tau2_method: str, inference_method: str
) -> None:
    effect = np.asarray([-0.4, 0.1, 0.5, 1.2, 1.8], dtype=np.float64)
    variance = np.asarray([0.04, 0.09, 0.05, 0.16, 0.08], dtype=np.float64)
    regression = fit_meta_regression(
        effect,
        variance,
        np.ones((len(effect), 1), dtype=np.float64),
        intercept=True,
        model="mixed",
        tau2_method=tau2_method,
        inference_method=inference_method,
        confidence_level=0.95,
        atol=1e-10,
        max_iter=1000,
    )
    pooled = ma.meta_analysis(
        effect=effect,
        variance=variance,
        model="random",
        tau2_method=tau2_method,
        ci_method=inference_method,
    )

    assert regression.tau2 is not None
    assert regression.tau2.value == pytest.approx(pooled.tau2, abs=1e-9)
    assert regression.coefficients[0] == pytest.approx(pooled.estimate, abs=1e-10)
    assert regression.standard_errors[0] == pytest.approx(
        pooled.standard_error, abs=1e-9
    )
    assert regression.ci_low[0] == pytest.approx(pooled.ci_low, abs=1e-9)
    assert regression.ci_high[0] == pytest.approx(pooled.ci_high, abs=1e-9)


def test_dataframe_categorical_encoding_is_explicit_and_replayable() -> None:
    data = _regression_frame()
    result = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["dose", "region"],
        categorical={"region": ["North", "South", "East"]},
        study="study",
        model="common",
    )

    assert result.design_info.term_names == (
        "intercept",
        "dose",
        "region[South]",
        "region[East]",
    )
    assert result.design_info.moderators[1].reference == "North"
    design = result.design_matrix
    assert design.loc["Study 1", "region[South]"] == 0.0
    assert design.loc["Study 2", "region[South]"] == 1.0
    assert design.loc["Study 3", "region[East]"] == 1.0
    assert result.test_moderator("region").terms == (
        "region[South]",
        "region[East]",
    )
    assert result.provenance.analysis_type == "meta_regression"
    assert result.provenance.column_mapping["moderator:dose"] == "dose"
    assert result.provenance.column_mapping["moderator:region"] == "region"
    assert any(
        transform.name == "categorical_treatment_coding"
        for transform in result.provenance.transformations
    )


def test_changing_categorical_reference_preserves_fitted_model() -> None:
    data = _regression_frame()
    north = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["North", "South", "East"]},
        model="mixed",
        tau2_method="PM",
    )
    south = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["South", "North", "East"]},
        model="mixed",
        tau2_method="PM",
    )

    np.testing.assert_allclose(
        north.study_results["fitted_value"], south.study_results["fitted_value"]
    )
    assert north.tau2 == pytest.approx(south.tau2)
    assert north.heterogeneity.q == pytest.approx(south.heterogeneity.q)
    assert north.global_test.statistic == pytest.approx(south.global_test.statistic)


def test_pm_solution_satisfies_residual_q_equation() -> None:
    data = _regression_frame()
    result = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["North", "South", "East"]},
        model="mixed",
        tau2_method="PM",
    )
    included = result.study_results.loc[result.study_results["included"]]
    design = result.design_matrix.to_numpy()
    weights = 1.0 / (included["variance"].to_numpy() + result.tau2)
    residual = (
        included["effect"].to_numpy()
        - design @ result.coefficients["estimate"].to_numpy()
    )

    assert float(np.dot(weights, residual * residual)) == pytest.approx(
        result.residual_df, rel=1e-9, abs=1e-9
    )


def test_reml_solution_satisfies_generalized_score() -> None:
    data = _regression_frame()
    result = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["North", "South", "East"]},
        model="mixed",
        tau2_method="REML",
    )
    included = result.study_results.loc[result.study_results["included"]]
    design = result.design_matrix.to_numpy()
    variance = included["variance"].to_numpy()
    effect = included["effect"].to_numpy()
    weights = 1.0 / (variance + result.tau2)
    gram_inverse = np.linalg.inv(design.T @ (weights[:, np.newaxis] * design))
    coefficients = gram_inverse @ design.T @ (weights * effect)
    residual = effect - design @ coefficients
    trace_p = np.sum(weights) - np.trace(
        gram_inverse @ design.T @ ((weights * weights)[:, np.newaxis] * design)
    )
    score = 0.5 * (np.dot(weights * weights, residual * residual) - trace_p)

    assert score == pytest.approx(0.0, abs=1e-8)


def test_generalized_dl_matches_moment_formula() -> None:
    data = _regression_frame()
    common = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["North", "South", "East"]},
        model="common",
    )
    mixed = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["North", "South", "East"]},
        model="mixed",
        tau2_method="DL",
    )
    design = common.design_matrix.to_numpy()
    variance = data["vi"].to_numpy()
    weights = 1.0 / variance
    inverse_gram = np.linalg.inv(design.T @ (weights[:, np.newaxis] * design))
    trace_p = np.sum(weights) - np.trace(
        inverse_gram @ design.T @ ((weights * weights)[:, np.newaxis] * design)
    )
    expected = max(0.0, (common.heterogeneity.q - common.residual_df) / trace_p)

    assert mixed.tau2 == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_hartung_knapp_covariance_and_global_f_test() -> None:
    data = _regression_frame()
    classic = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["North", "South", "East"]},
        model="mixed",
        tau2_method="REML",
        inference_method="normal",
    )
    hk = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["region"],
        categorical={"region": ["North", "South", "East"]},
        model="mixed",
        tau2_method="REML",
        inference_method="hartung_knapp",
    )

    assert hk.tau2 == pytest.approx(classic.tau2)
    np.testing.assert_allclose(
        hk.coefficient_covariance,
        hk.diagnostics.residual_scale * classic.coefficient_covariance,
    )
    assert hk.global_test.distribution == "F"
    assert hk.global_test.df_num == 2
    assert hk.global_test.df_denom == hk.residual_df
    assert set(hk.coefficients["statistic_name"]) == {"t"}
    assert set(hk.coefficients["df"]) == {float(hk.residual_df)}


def test_hartung_knapp_adhoc_never_reduces_classic_covariance() -> None:
    data = pd.DataFrame(
        {
            "yi": [0.10, 0.11, 0.12, 0.13, 0.14],
            "vi": [0.04] * 5,
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    classic = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["x"],
        model="mixed",
        inference_method="normal",
    )
    protected = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["x"],
        model="mixed",
        inference_method="hartung_knapp_adhoc",
    )

    np.testing.assert_allclose(
        protected.coefficient_covariance, classic.coefficient_covariance
    )
    assert protected.diagnostics.residual_scale == 1.0


def test_prediction_replays_training_design_and_rejects_unknown_category() -> None:
    data = _regression_frame()
    result = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["dose", "region"],
        categorical={"region": ["North", "South", "East"]},
        model="mixed",
    )
    prediction = result.predict(data[["dose", "region"]])

    np.testing.assert_allclose(
        prediction["estimate"], result.study_results["fitted_value"]
    )
    assert {"pi_low", "pi_high"}.issubset(prediction.columns)
    with pytest.raises(ma.InvalidStudyDataError, match="unknown levels"):
        result.predict(pd.DataFrame({"dose": [1.0], "region": ["West"]}))
    with pytest.raises(ma.InvalidStudyDataError, match="missing moderator column"):
        result.predict(pd.DataFrame({"dose": [1.0]}))


def test_common_prediction_omits_prediction_interval() -> None:
    result = ma.meta_regression(
        effect=[0.1, 0.4, 0.9, 1.3],
        variance=[0.04, 0.05, 0.06, 0.07],
        moderators={"x": [0.0, 1.0, 2.0, 3.0]},
        model="common",
    )

    prediction = result.predict(pd.DataFrame({"x": [1.5]}))
    assert "pi_low" not in prediction
    assert "pi_high" not in prediction


def test_missing_drop_combines_reasons_and_preserves_excluded_rows() -> None:
    data = pd.DataFrame(
        {
            "study": ["A", "B", None, "D", "E", "F"],
            "yi": [0.1, np.nan, 0.4, 0.7, 0.9, 1.1],
            "vi": [0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            "x": [0.0, np.nan, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["x"],
        study="study",
        model="common",
        missing="drop",
    )

    assert result.k == 4
    assert result.study_results["exclusion_reason"].tolist() == [
        None,
        "missing effect and moderator 'x'",
        "missing study",
        None,
        None,
        None,
    ]
    assert result.study_results.loc[1, "included"] == np.False_
    assert np.isnan(result.study_results.loc[1, "fitted_value"])
    assert np.isnan(result.study_results.loc[2, "precision_weight"])
    assert result.provenance.excluded_rows == (1, 2)


def test_missing_raise_reports_moderator_fields() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="moderator 'x'"):
        ma.meta_regression(
            effect=[0.1, 0.2, 0.3],
            variance=[0.04, 0.05, 0.06],
            moderators={"x": [0.0, np.nan, 2.0]},
            model="common",
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"moderators": []}, "At least one moderator"),
        ({"moderators": "x"}, "sequence of column names"),
        (
            {"moderators": {"group": ["a", "b", "a"]}},
            "declare it in categorical",
        ),
        (
            {
                "moderators": {"group": ["a", "b", "c"]},
                "categorical": {"group": ["a", "b"]},
            },
            "undeclared levels",
        ),
        (
            {
                "moderators": {"x": [0.0, 1.0, 2.0]},
                "categorical": {"group": ["a", "b"]},
            },
            "unknown moderator",
        ),
    ],
)
def test_invalid_moderator_declarations_are_explicit(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match=match):
        ma.meta_regression(
            effect=[0.1, 0.2, 0.3],
            variance=[0.04, 0.05, 0.06],
            model="common",
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "name",
    [
        "intercept",
        "study",
        "effect",
        "variance",
        "included",
        "fitted_value",
        "precision_weight",
        "leverage",
    ],
)
def test_result_column_names_are_reserved_for_moderators(name: str) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="reserved"):
        ma.meta_regression(
            effect=[0.1, 0.2, 0.3],
            variance=[0.04, 0.05, 0.06],
            moderators={name: [0.0, 1.0, 2.0]},
            model="common",
        )


def test_absent_declared_level_after_exclusion_is_rejected() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="absent after exclusions"):
        ma.meta_regression(
            effect=[0.1, 0.2, np.nan, 0.4],
            variance=[0.04, 0.05, 0.06, 0.07],
            moderators={"group": ["a", "b", "c", "a"]},
            categorical={"group": ["a", "b", "c"]},
            model="common",
            missing="drop",
        )


def test_rank_deficiency_and_insufficient_residual_df_are_rejected() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="rank deficient"):
        ma.meta_regression(
            effect=[0.1, 0.2, 0.3, 0.4],
            variance=[0.04] * 4,
            moderators={"x": [0, 1, 2, 3], "copy": [0, 1, 2, 3]},
            model="common",
        )
    with pytest.raises(ma.InsufficientStudiesError, match="k > p"):
        ma.meta_regression(
            effect=[0.1, 0.2],
            variance=[0.04, 0.05],
            moderators={"x": [0, 1]},
            model="common",
        )


def test_unsupported_model_and_inference_combinations_are_rejected() -> None:
    with pytest.raises(ma.UnsupportedMethodError, match="Unsupported model"):
        ma.meta_regression(
            effect=[0.1, 0.2, 0.3],
            variance=[0.04, 0.05, 0.06],
            moderators={"x": [0, 1, 2]},
            model="mystery",
        )
    with pytest.raises(ma.UnsupportedMethodError, match="only supported"):
        ma.meta_regression(
            effect=[0.1, 0.2, 0.3],
            variance=[0.04, 0.05, 0.06],
            moderators={"x": [0, 1, 2]},
            model="common",
            inference_method="hartung_knapp",
        )
    with pytest.raises(ma.UnsupportedMethodError, match="tau2_method"):
        ma.meta_regression(
            effect=[0.1, 0.2, 0.3],
            variance=[0.04, 0.05, 0.06],
            moderators={"x": [0, 1, 2]},
            model="mixed",
            tau2_method="mystery",
        )


def test_no_intercept_numeric_model_is_supported_but_categorical_is_not() -> None:
    result = ma.meta_regression(
        effect=[0.1, 0.4, 0.9],
        variance=[0.04, 0.05, 0.06],
        moderators={"x": [1.0, 2.0, 3.0]},
        model="common",
        intercept=False,
    )
    assert result.design_info.term_names == ("x",)
    assert result.pseudo_r2 is None

    with pytest.raises(ma.InvalidStudyDataError, match="only supported with numeric"):
        ma.meta_regression(
            effect=[0.1, 0.4, 0.9],
            variance=[0.04, 0.05, 0.06],
            moderators={"group": ["a", "b", "a"]},
            categorical={"group": ["a", "b"]},
            model="common",
            intercept=False,
        )


def test_result_is_defensive_and_has_no_ambiguous_pooled_estimate() -> None:
    result = ma.meta_regression(
        effect=[0.1, 0.4, 0.9, 1.2],
        variance=[0.04, 0.05, 0.06, 0.07],
        moderators={"x": [0.0, 1.0, 2.0, 3.0]},
        model="common",
    )
    coefficients = result.coefficients
    coefficients.loc[0, "estimate"] = 999.0
    covariance = result.coefficient_covariance
    covariance.iloc[0, 0] = 999.0
    studies = result.study_results
    studies.loc[0, "effect"] = 999.0

    assert result.coefficients.loc[0, "estimate"] != 999.0
    assert result.coefficient_covariance.iloc[0, 0] != 999.0
    assert result.study_results.loc[0, "effect"] != 999.0
    assert not hasattr(result, "estimate")
    with pytest.raises(FrozenInstanceError):
        result.tau2 = 3.0  # type: ignore[misc]


def test_report_and_summary_include_auditable_regression_details() -> None:
    data = _regression_frame()
    result = ma.meta_regression(
        data,
        effect="yi",
        standard_error=np.sqrt(data["vi"]),
        moderators=["dose"],
        model="mixed",
    )
    payload = result.report(include_studies=False).to_dict()

    assert payload["schema_version"] == "1.2"
    assert payload["report_type"] == "meta_regression"
    assert "studies" not in payload
    assert payload["design"]["term_names"] == ["intercept", "dose"]
    assert payload["global_moderator_test"]["distribution"] == "chi_square"
    assert "study-level associations" in result.method_details()
    assert "individual-level causal effects" in str(result.summary())
    assert result.provenance.transformations[0].name == "standard_error_to_variance"


def test_model_and_inference_aliases_are_normalized() -> None:
    result = ma.meta_regression(
        effect=[0.1, 0.4, 0.9, 1.2],
        variance=[0.04, 0.05, 0.06, 0.07],
        moderators={"x": [0.0, 1.0, 2.0, 3.0]},
        model="random-effects",
        inference_method="hk",
    )

    assert result.model == "mixed"
    assert result.method.inference_method == "hartung_knapp"


def test_centering_moderator_changes_intercept_not_slope_or_fit() -> None:
    data = _regression_frame()
    centered_data = data.assign(centered=data["dose"] - data["dose"].mean())
    original = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["dose"],
        model="mixed",
    )
    centered = ma.meta_regression(
        centered_data,
        effect="yi",
        variance="vi",
        moderators=["centered"],
        model="mixed",
    )

    assert original.coefficients.loc[1, "estimate"] == pytest.approx(
        centered.coefficients.loc[1, "estimate"]
    )
    np.testing.assert_allclose(
        original.study_results["fitted_value"], centered.study_results["fitted_value"]
    )
    assert original.tau2 == pytest.approx(centered.tau2)
