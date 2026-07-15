"""Binary meta-analysis tests with statsmodels and RevMan reference values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma

EVENT_TREAT = np.array([12, 5, 20, 7])
N_TREAT = np.array([100, 80, 120, 90])
EVENT_CONTROL = np.array([18, 9, 15, 10])
N_CONTROL = np.array([110, 75, 130, 95])


@pytest.mark.parametrize(
    ("measure", "expected_effect", "expected_variance", "pooled", "se"),
    [
        (
            "OR",
            [-0.36101335, -0.71562004, 0.42744401, -0.33286430],
            [0.16112209, 0.33959596, 0.13536232, 0.26667004],
            -0.12018213361646543,
            0.2220102776766474,
        ),
        (
            "RR",
            [-0.31015493, -0.65232519, 0.36772478, -0.30260772],
            [0.11979798, 0.28527778, 0.10064103, 0.22121972],
            -0.09843214012915684,
            0.1949594080038293,
        ),
        (
            "RD",
            [-0.04363636, -0.0575, 0.05128205, -0.02748538],
            [0.00230018, 0.00214042, 0.00194257, 0.00178838],
            -0.01761456285145937,
            0.022496666279404356,
        ),
    ],
)
def test_inverse_variance_effects_match_statsmodels_reference(
    measure: str,
    expected_effect: list[float],
    expected_variance: list[float],
    pooled: float,
    se: float,
) -> None:
    result = ma.meta_binary(
        event_treat=EVENT_TREAT,
        n_treat=N_TREAT,
        event_control=EVENT_CONTROL,
        n_control=N_CONTROL,
        measure=measure,
        method="IV",
        model="common",
    )

    np.testing.assert_allclose(
        result.study_results["effect"], expected_effect, atol=5e-9
    )
    np.testing.assert_allclose(
        result.study_results["variance"], expected_variance, atol=5e-9
    )
    assert result.estimate == pytest.approx(pooled, abs=1e-13)
    assert result.standard_error == pytest.approx(se, abs=1e-13)
    assert result.method.pooling_method == "inverse_variance"


def test_mantel_haenszel_or_matches_statsmodels_stratified_table() -> None:
    result = ma.meta_binary(
        event_treat=EVENT_TREAT,
        n_treat=N_TREAT,
        event_control=EVENT_CONTROL,
        n_control=N_CONTROL,
        measure="OR",
        method="MH",
    )

    assert result.estimate == pytest.approx(-0.12255185912117174, abs=1e-13)
    assert result.display_estimate == pytest.approx(0.8846600260623873, abs=1e-13)
    assert result.standard_error == pytest.approx(0.2182290588355789, abs=1e-13)
    assert result.method.pooling_method == "mantel_haenszel"
    assert result.effect_scale == "log"
    assert result.display_scale == "exp"
    assert "Estimate: 0.88466" in str(result.summary())
    assert result.summary().to_dict()["pooling_method"] == "mantel_haenszel"


def test_mantel_haenszel_rr_matches_revman_formula_reference() -> None:
    result = ma.meta_binary(
        event_treat=EVENT_TREAT,
        n_treat=N_TREAT,
        event_control=EVENT_CONTROL,
        n_control=N_CONTROL,
        measure="RR",
        method="MH",
    )

    assert result.estimate == pytest.approx(-0.10772100195106375, abs=1e-13)
    assert result.display_estimate == pytest.approx(0.8978780677173858, abs=1e-13)
    assert result.standard_error == pytest.approx(0.1915886270134988, abs=1e-13)
    assert result.display_ci == pytest.approx((0.6167892961382874, 1.3070671451911116))
    assert result.study_results["normalized_weight"].sum() == pytest.approx(1.0)


def test_dataframe_input_defaults_to_index_and_supports_random_iv() -> None:
    data = pd.DataFrame(
        {
            "et": EVENT_TREAT,
            "nt": N_TREAT,
            "ec": EVENT_CONTROL,
            "nc": N_CONTROL,
        },
        index=["A", "B", "C", "D"],
    )
    result = ma.meta_binary(
        data,
        event_treat="et",
        n_treat="nt",
        event_control="ec",
        n_control="nc",
        measure="RR",
        method="IV",
        model="random",
        tau2_method="PM",
    )

    assert result.study_results["study"].tolist() == ["A", "B", "C", "D"]
    assert result.model == "random"
    assert result.method.tau2_method == "PM"
    assert result.prediction_interval is not None
    assert result.display_prediction_interval is not None


def test_single_zero_is_corrected_for_effect_but_not_exact_mh_pooling() -> None:
    result = ma.meta_binary(
        event_treat=[0, 8, 12],
        n_treat=[50, 60, 70],
        event_control=[4, 6, 10],
        n_control=[50, 60, 70],
        measure="RR",
        method="MH",
    )

    assert result.study_results["continuity_corrected"].tolist() == [True, False, False]
    assert result.study_results["mh_continuity_corrected"].tolist() == [
        False,
        False,
        False,
    ]
    assert np.isfinite(result.study_results.loc[0, "effect"])
    assert any("individual effects" in warning for warning in result.warnings)


def test_mh_correction_is_explicit_when_exact_estimator_is_undefined() -> None:
    kwargs = {
        "event_treat": [0, 0],
        "n_treat": [40, 50],
        "event_control": [2, 3],
        "n_control": [40, 50],
        "measure": "RR",
        "method": "MH",
    }
    with pytest.raises(ma.InvalidStudyDataError, match="exact Mantel-Haenszel RR"):
        ma.meta_binary(**kwargs)

    corrected = ma.meta_binary(**kwargs, mh_continuity_correction=0.5)
    assert corrected.study_results["mh_continuity_corrected"].all()
    assert np.isfinite(corrected.estimate)
    assert any("MH pooling" in warning for warning in corrected.warnings)


def test_exact_mh_or_reports_zero_cross_product() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="exact Mantel-Haenszel OR"):
        ma.meta_binary(
            event_treat=[0, 0],
            n_treat=[40, 50],
            event_control=[2, 3],
            n_control=[40, 50],
            measure="OR",
            method="MH",
        )


def test_double_zero_and_double_all_are_excluded_for_relative_effects() -> None:
    result = ma.meta_binary(
        event_treat=[0, 20, 4],
        n_treat=[20, 20, 20],
        event_control=[0, 20, 5],
        n_control=[20, 20, 20],
        measure="OR",
        method="IV",
        model="common",
    )

    studies = result.study_results
    assert result.k == 1
    assert studies["included"].tolist() == [False, False, True]
    assert studies.loc[0, "exclusion_reason"] == "no events in either group"
    assert "all participants" in studies.loc[1, "exclusion_reason"]
    assert np.isnan(studies.loc[0, "weight"])
    assert result.q_df == 0


def test_double_zero_is_included_for_risk_difference_with_variance_correction() -> None:
    result = ma.meta_binary(
        event_treat=[0, 4],
        n_treat=[20, 20],
        event_control=[0, 5],
        n_control=[20, 20],
        measure="RD",
        method="IV",
        model="common",
    )

    studies = result.study_results
    assert studies["included"].tolist() == [True, True]
    assert studies.loc[0, "effect"] == 0.0
    assert studies.loc[0, "variance"] > 0.0
    assert bool(studies.loc[0, "continuity_corrected"])


def test_zero_correction_can_be_disabled_only_when_effect_remains_defined() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="remaining zero cells"):
        ma.meta_binary(
            event_treat=[0, 4],
            n_treat=[20, 20],
            event_control=[2, 5],
            n_control=[20, 20],
            measure="OR",
            method="IV",
            model="common",
            continuity_correction=0.0,
            correction_scope="none",
        )

    with pytest.raises(ma.InvalidStudyDataError, match="Non-positive binary"):
        ma.meta_binary(
            event_treat=[0, 4],
            n_treat=[20, 20],
            event_control=[0, 5],
            n_control=[20, 20],
            measure="RD",
            method="IV",
            model="common",
            continuity_correction=0.0,
            correction_scope="none",
        )


def test_correction_scope_controls_which_studies_are_adjusted() -> None:
    base = {
        "event_treat": [0, 4],
        "n_treat": [20, 20],
        "event_control": [2, 5],
        "n_control": [20, 20],
        "measure": "RR",
        "method": "IV",
        "model": "common",
    }
    only_zero = ma.meta_binary(**base, correction_scope="only0")
    if_any = ma.meta_binary(**base, correction_scope="if0all")
    all_studies = ma.meta_binary(**base, correction_scope="all")

    assert only_zero.study_results["continuity_corrected"].tolist() == [True, False]
    assert if_any.study_results["continuity_corrected"].tolist() == [True, True]
    assert all_studies.study_results["continuity_corrected"].tolist() == [True, True]

    no_zero = ma.meta_binary(
        event_treat=[1, 4],
        n_treat=[20, 20],
        event_control=[2, 5],
        n_control=[20, 20],
        measure="RR",
        method="IV",
        model="common",
        correction_scope="if_any_zero",
    )
    assert not no_zero.study_results["continuity_corrected"].any()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"event_treat": [11], "n_treat": [10]}, "between 0 and n_treat"),
        ({"event_treat": [1.5], "n_treat": [10]}, "whole-number counts"),
        ({"event_treat": [0], "n_treat": [0]}, "strictly positive"),
    ],
)
def test_invalid_binary_counts_raise_domain_errors(
    kwargs: dict[str, list[float]], match: str
) -> None:
    complete = {
        "event_treat": [1],
        "n_treat": [10],
        "event_control": [2],
        "n_control": [10],
    }
    complete.update(kwargs)
    with pytest.raises(ma.InvalidStudyDataError, match=match):
        ma.meta_binary(**complete, measure="RR", method="IV", model="common")


def test_missing_binary_row_is_retained_with_reason() -> None:
    result = ma.meta_binary(
        event_treat=[1, np.nan, 3],
        n_treat=[10, 10, 10],
        event_control=[2, 2, 2],
        n_control=[10, np.nan, 10],
        measure="RR",
        method="IV",
        model="common",
        missing="drop",
    )

    assert result.k == 2
    assert result.study_results.loc[1, "exclusion_reason"] == (
        "missing event_treat, n_control"
    )


def test_all_missing_or_all_uninformative_binary_data_raise() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="No studies remain"):
        ma.meta_binary(
            event_treat=[np.nan],
            n_treat=[np.nan],
            event_control=[np.nan],
            n_control=[np.nan],
            missing="drop",
            method="IV",
            model="common",
        )

    with pytest.raises(ma.InvalidStudyDataError, match="No informative studies"):
        ma.meta_binary(
            event_treat=[0, 10],
            n_treat=[10, 10],
            event_control=[0, 10],
            n_control=[10, 10],
            measure="RR",
            method="IV",
            model="common",
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"continuity_correction": -0.5}, "finite and non-negative"),
        ({"continuity_correction": "half"}, "non-negative number"),
        ({"correction_scope": "mystery"}, "correction_scope must be"),
    ],
)
def test_invalid_continuity_correction_options(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(
        (ma.InvalidStudyDataError, ma.UnsupportedMethodError), match=match
    ):
        ma.meta_binary(
            event_treat=[1, 2],
            n_treat=[10, 10],
            event_control=[2, 3],
            n_control=[10, 10],
            method="IV",
            model="common",
            **kwargs,  # type: ignore[arg-type]
        )


def test_binary_vectors_must_have_equal_lengths() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="equal lengths"):
        ma.meta_binary(
            event_treat=[1, 2],
            n_treat=[10],
            event_control=[2, 3],
            n_control=[10, 10],
            method="IV",
            model="common",
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "MH", "model": "random"}, "only for model='common'"),
        ({"method": "MH", "measure": "RD"}, "use method='IV'"),
        ({"method": "MH", "ci_method": "HK"}, "only ci_method='normal'"),
        ({"method": "mystery"}, "method must be"),
        ({"measure": "mystery", "method": "IV"}, "measure must be"),
    ],
)
def test_unsupported_binary_method_combinations(
    kwargs: dict[str, str], match: str
) -> None:
    base = {
        "event_treat": [1, 2],
        "n_treat": [10, 10],
        "event_control": [2, 3],
        "n_control": [10, 10],
    }
    with pytest.raises(ma.UnsupportedMethodError, match=match):
        ma.meta_binary(**base, **kwargs)


@pytest.mark.parametrize("measure", ["OR", "RR"])
@pytest.mark.parametrize("method", ["IV", "MH"])
def test_swapping_groups_inverts_relative_pooled_effect(
    measure: str, method: str
) -> None:
    forward = ma.meta_binary(
        event_treat=EVENT_TREAT,
        n_treat=N_TREAT,
        event_control=EVENT_CONTROL,
        n_control=N_CONTROL,
        measure=measure,
        method=method,
        model="common",
    )
    reverse = ma.meta_binary(
        event_treat=EVENT_CONTROL,
        n_treat=N_CONTROL,
        event_control=EVENT_TREAT,
        n_control=N_TREAT,
        measure=measure,
        method=method,
        model="common",
    )

    assert reverse.display_estimate == pytest.approx(1.0 / forward.display_estimate)
    assert reverse.display_ci == pytest.approx(
        (1.0 / forward.display_ci[1], 1.0 / forward.display_ci[0])
    )
