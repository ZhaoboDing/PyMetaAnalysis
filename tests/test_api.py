from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma


def test_common_effect_matches_hand_calculation() -> None:
    result = ma.meta_analysis(
        effect=[1.0, 2.0, 3.0],
        variance=[1.0, 1.0, 1.0],
        model="common",
    )

    assert result.model == "common"
    assert result.estimate == pytest.approx(2.0)
    assert result.standard_error == pytest.approx(np.sqrt(1.0 / 3.0))
    assert result.q == pytest.approx(2.0)
    assert result.q_df == 2
    assert result.i2 == pytest.approx(0.0)
    assert result.h2 == pytest.approx(1.0)
    assert result.tau2 == 0.0
    assert result.method.tau2_method is None
    assert result.prediction_interval is None
    assert result.study_results["normalized_weight"].sum() == pytest.approx(1.0)


def test_dataframe_columns_and_default_index_labels() -> None:
    data = pd.DataFrame(
        {"yi": [0.1, 0.4, -0.2], "vi": [0.04, 0.09, 0.16]},
        index=pd.Index(["Alpha", "Beta", "Gamma"], name="trial"),
    )

    result = ma.meta_analysis(data, effect="yi", variance="vi", model="fixed")

    assert result.model == "common"
    assert result.study_results["study"].tolist() == ["Alpha", "Beta", "Gamma"]
    assert result.study_results["row_id"].tolist() == [0, 1, 2]


def test_standard_error_input_matches_sampling_variance_input() -> None:
    effect = np.asarray([0.1, 0.4, -0.2])
    standard_error = np.asarray([0.2, 0.3, 0.4])

    from_standard_error = ma.meta_analysis(
        effect=effect,
        standard_error=standard_error,
        model="common",
    )
    from_variance = ma.meta_analysis(
        effect=effect,
        variance=standard_error**2,
        model="common",
    )

    assert from_standard_error.estimate == pytest.approx(from_variance.estimate)
    assert from_standard_error.standard_error == pytest.approx(
        from_variance.standard_error
    )
    assert from_standard_error.q == pytest.approx(from_variance.q)
    np.testing.assert_allclose(
        from_standard_error.study_results["variance"], standard_error**2
    )
    np.testing.assert_allclose(
        from_standard_error.study_results["standard_error"], standard_error
    )


def test_explicit_study_column_overrides_dataframe_index() -> None:
    data = pd.DataFrame(
        {
            "label": ["A", "B"],
            "yi": [0.1, 0.2],
            "vi": [0.01, 0.02],
        },
        index=[10, 20],
    )

    result = ma.meta_analysis(
        data,
        effect="yi",
        variance="vi",
        study="label",
        model="common",
    )

    assert result.study_results["study"].tolist() == ["A", "B"]


def test_missing_drop_is_visible_and_excluded_from_weights() -> None:
    result = ma.meta_analysis(
        effect=[0.1, np.nan, 0.4],
        variance=[0.01, 0.02, 0.03],
        model="common",
        missing="drop",
    )

    studies = result.study_results
    assert result.k == 2
    assert studies["included"].tolist() == [True, False, True]
    assert studies.loc[1, "exclusion_reason"] == "missing effect"
    assert np.isnan(studies.loc[1, "weight"])
    assert result.excluded_studies["row_id"].tolist() == [1]
    assert any("Excluded 1 study" in warning for warning in result.warnings)


def test_result_table_is_returned_as_a_defensive_copy() -> None:
    result = ma.meta_analysis(effect=[0.1, 0.2], variance=[0.01, 0.02], model="common")
    first = result.study_results
    first.loc[0, "effect"] = 999.0

    assert result.study_results.loc[0, "effect"] == pytest.approx(0.1)
    with pytest.raises(FrozenInstanceError):
        result.estimate = 4.0  # type: ignore[misc]


def test_summary_has_text_and_machine_readable_forms() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.2, 0.3], variance=[0.01, 0.02, 0.04], model="common"
    )

    rendered = str(result.summary())
    values = result.summary().to_dict()

    assert "Meta-analysis (common-effect, GENERIC)" in rendered
    assert "I^2 (Q-based):" in rendered
    assert values["estimate"] == result.estimate
    assert values["studies"] == 3
    assert values["i2_method"] == "q_based"
    assert result.ci == (result.ci_low, result.ci_high)
    pd.testing.assert_frame_equal(result.to_dataframe(), result.study_results)


def test_single_study_common_model_reports_unavailable_heterogeneity() -> None:
    result = ma.meta_analysis(effect=[0.5], variance=[0.04], model="common")

    assert result.q_df == 0
    assert np.isnan(result.q_pvalue)
    assert np.isnan(result.i2)
    assert "not estimable with one study" in str(result.summary())


def test_random_effects_i2_uses_tau2_and_typical_within_study_variance() -> None:
    variance = np.asarray([0.02, 0.03, 0.04, 0.08], dtype=np.float64)
    result = ma.meta_analysis(
        effect=[-0.7, 0.0, 0.8, 1.5],
        variance=variance,
        model="random",
        tau2_method="REML",
    )
    weights = 1.0 / variance
    c_value = weights.sum() - np.dot(weights, weights) / weights.sum()
    typical_variance = (len(variance) - 1) / c_value

    assert result.i2_method == "tau2_typical_variance"
    assert result.i2 == pytest.approx(result.tau2 / (result.tau2 + typical_variance))
    assert result.h2 == pytest.approx(1.0 + result.tau2 / typical_variance)
    assert "tau^2/typical-variance" in str(result.summary())


def test_fixed_alias_and_ci_aliases_are_resolved() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.2],
        variance=[0.01, 0.02],
        model="fixed-effect",
        ci_method="z",
    )

    assert result.method.model == "common"
    assert result.method.ci_method == "normal"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"effect": [1.0], "variance": [1.0, 2.0]}, "same length"),
        ({"effect": [1.0], "variance": [0.0]}, "strictly positive"),
        ({"effect": [np.inf], "variance": [1.0]}, "must be finite"),
        ({"effect": [np.nan], "variance": [1.0]}, "Missing effect"),
    ],
)
def test_invalid_study_data_raises_domain_error(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match=match):
        ma.meta_analysis(**kwargs, model="common")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"effect": [0.1]}, "Exactly one"),
        (
            {
                "effect": [0.1],
                "variance": [0.01],
                "standard_error": [0.1],
            },
            "Exactly one",
        ),
        (
            {"effect": [0.1], "standard_error": [0.0]},
            "standard errors must be strictly positive",
        ),
        (
            {"effect": [0.1], "standard_error": [np.inf]},
            "Standard error values must be finite",
        ),
        (
            {"effect": [0.1], "standard_error": ["invalid"]},
            "standard error must contain numeric values",
        ),
        (
            {"effect": [0.1], "standard_error": [1e308]},
            "finite, strictly positive sampling variances after squaring",
        ),
        (
            {"effect": [0.1], "standard_error": [1e-300]},
            "finite, strictly positive sampling variances after squaring",
        ),
    ],
)
def test_standard_error_input_errors_are_explicit(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match=match):
        ma.meta_analysis(**kwargs, model="common")  # type: ignore[arg-type]


def test_missing_standard_error_drop_records_specific_reasons() -> None:
    result = ma.meta_analysis(
        effect=[np.nan, 0.2, 0.3],
        standard_error=[np.nan, np.nan, 0.2],
        missing="drop",
        model="common",
    )

    assert result.study_results["exclusion_reason"].tolist() == [
        "missing effect and standard error",
        "missing standard error",
        None,
    ]


def test_random_effects_requires_two_included_studies() -> None:
    with pytest.raises(ma.InsufficientStudiesError, match="at least two"):
        ma.meta_analysis(effect=[0.1], variance=[0.01], model="random")


def test_unknown_methods_raise_domain_errors() -> None:
    with pytest.raises(ma.UnsupportedMethodError, match="Unsupported model"):
        ma.meta_analysis(effect=[0.1], variance=[0.01], model="mystery")

    with pytest.raises(ma.UnsupportedMethodError, match="tau2_method"):
        ma.meta_analysis(
            effect=[0.1, 0.3],
            variance=[0.01, 0.02],
            model="random",
            tau2_method="mystery",
        )

    with pytest.raises(ma.UnsupportedMethodError, match="ci_method"):
        ma.meta_analysis(
            effect=[0.1, 0.3],
            variance=[0.01, 0.02],
            model="random",
            ci_method="mystery",
        )


@pytest.mark.parametrize(
    ("parameter", "value", "match"),
    [
        ("confidence_level", 1.0, "between 0 and 1"),
        ("max_iter", 0, "positive integer"),
        ("atol", 0.0, "strictly positive"),
    ],
)
def test_invalid_numerical_controls_raise_domain_errors(
    parameter: str, value: object, match: str
) -> None:
    kwargs = {parameter: value}
    with pytest.raises(ma.InvalidStudyDataError, match=match):
        ma.meta_analysis(
            effect=[0.1, 0.2],
            variance=[0.01, 0.02],
            model="common",
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"effect": "yi", "variance": [0.1]}, "no DataFrame"),
        (
            {
                "data": pd.DataFrame({"yi": [1.0]}),
                "effect": "missing",
                "variance": [0.1],
            },
            "not present",
        ),
        ({"effect": [[1.0, 2.0]], "variance": [0.1, 0.2]}, "one-dimensional"),
        (
            {"effect": [1.0, 2.0], "variance": [0.1, 0.2], "study": ["A"]},
            "study has length",
        ),
        (
            {"data": [1, 2], "effect": [1.0, 2.0], "variance": [0.1, 0.2]},
            "pandas DataFrame",
        ),
        (
            {"effect": [1.0, 2.0], "variance": [0.1, 0.2], "missing": "ignore"},
            "missing must be",
        ),
        ({"effect": ["a", "b"], "variance": [0.1, 0.2]}, "numeric values"),
        ({"effect": [1.0], "variance": [np.inf]}, "Variance values must be finite"),
    ],
)
def test_input_shape_and_resolution_errors(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ma.InvalidStudyDataError, match=match):
        ma.meta_analysis(**kwargs, model="common")  # type: ignore[arg-type]


def test_array_inputs_with_dataframe_must_match_frame_length() -> None:
    data = pd.DataFrame({"unused": [1, 2, 3]})
    with pytest.raises(ma.InvalidStudyDataError, match="one value per DataFrame row"):
        ma.meta_analysis(
            data,
            effect=[0.1, 0.2],
            variance=[0.01, 0.02],
            model="common",
        )


def test_missing_reason_variants_and_all_dropped_error() -> None:
    result = ma.meta_analysis(
        effect=[np.nan, 0.2, 0.3],
        variance=[np.nan, np.nan, 0.02],
        missing="drop",
        model="common",
    )
    assert result.study_results["exclusion_reason"].tolist() == [
        "missing effect and variance",
        "missing variance",
        None,
    ]

    with pytest.raises(ma.InvalidStudyDataError, match="No studies remain"):
        ma.meta_analysis(
            effect=[np.nan],
            variance=[np.nan],
            missing="drop",
            model="common",
        )
