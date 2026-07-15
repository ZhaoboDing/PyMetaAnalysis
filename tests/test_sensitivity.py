from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma


def test_leave_one_out_common_effect_matches_hand_calculation() -> None:
    result = ma.meta_analysis(
        effect=[1.0, 2.0, 4.0],
        variance=[1.0, 1.0, 1.0],
        study=["A", "B", "C"],
        model="common",
    )
    influence = result.leave_one_out()
    table = influence.to_dataframe()

    assert isinstance(influence, ma.LeaveOneOutResult)
    assert len(influence) == 3
    assert table["omitted_row_id"].tolist() == [0, 1, 2]
    assert table["omitted_study"].tolist() == ["A", "B", "C"]
    np.testing.assert_allclose(table["estimate"], [3.0, 2.5, 1.5])
    np.testing.assert_allclose(table["standard_error"], np.sqrt(0.5))
    assert table["k"].tolist() == [2, 2, 2]


def test_leave_one_out_preserves_random_effects_method_configuration() -> None:
    result = ma.meta_analysis(
        effect=[-0.2, 0.1, 0.8, 1.4],
        variance=[0.03, 0.05, 0.04, 0.06],
        model="random",
        tau2_method="PM",
        ci_method="hartung_knapp_adhoc",
        confidence_level=0.9,
        atol=1e-8,
        max_iter=321,
    )
    influence = result.leave_one_out()

    assert len(influence) == 4
    for refit in influence.results:
        assert refit.model == "random"
        assert refit.method.tau2_method == "PM"
        assert refit.method.ci_method == "hartung_knapp_adhoc"
        assert refit.method.confidence_level == pytest.approx(0.9)
        assert refit.method.atol == pytest.approx(1e-8)
        assert refit.method.max_iter == 321


def test_mantel_haenszel_leave_one_out_matches_direct_refits() -> None:
    event_treat = np.array([12, 5, 20, 7])
    n_treat = np.array([100, 80, 120, 90])
    event_control = np.array([18, 9, 15, 10])
    n_control = np.array([110, 75, 130, 95])
    result = ma.meta_binary(
        event_treat=event_treat,
        n_treat=n_treat,
        event_control=event_control,
        n_control=n_control,
        measure="RR",
        method="MH",
    )

    influence = result.leave_one_out()
    for omitted, refit in enumerate(influence.results):
        keep = np.arange(4) != omitted
        direct = ma.meta_binary(
            event_treat=event_treat[keep],
            n_treat=n_treat[keep],
            event_control=event_control[keep],
            n_control=n_control[keep],
            measure="RR",
            method="MH",
        )
        assert refit.estimate == pytest.approx(direct.estimate, abs=1e-14)
        assert refit.standard_error == pytest.approx(direct.standard_error, abs=1e-14)
        assert refit.method.pooling_method == "mantel_haenszel"


def test_risk_difference_leave_one_out_preserves_zero_variance_policy() -> None:
    result = ma.meta_binary(
        event_treat=[0, 4, 6, 8],
        n_treat=[20, 20, 20, 20],
        event_control=[0, 5, 4, 7],
        n_control=[20, 20, 20, 20],
        measure="RD",
        method="IV",
        model="common",
        rd_zero_variance="exclude",
    )

    assert result.k == 3
    influence = result.leave_one_out()
    assert len(influence) == 3
    for refit in influence.results:
        assert dict(refit.method.options)["rd_zero_variance"] == "exclude"


def test_smd_leave_one_out_recomputes_effects_from_raw_inputs() -> None:
    inputs = {
        "mean_treat": np.array([4.0, 5.0, 6.2]),
        "sd_treat": np.array([1.2, 1.4, 1.1]),
        "n_treat": np.array([20, 25, 30]),
        "mean_control": np.array([3.5, 4.2, 5.0]),
        "sd_control": np.array([1.1, 1.3, 1.2]),
        "n_control": np.array([18, 24, 28]),
    }
    result = ma.meta_continuous(
        **inputs,
        measure="SMD",
        model="common",
    )
    influence = result.leave_one_out()

    for omitted, refit in enumerate(influence.results):
        keep = np.arange(3) != omitted
        direct = ma.meta_continuous(
            **{name: values[keep] for name, values in inputs.items()},
            measure="SMD",
            model="common",
        )
        assert refit.estimate == pytest.approx(direct.estimate, rel=1e-14)
        assert dict(refit.method.options)["effect_estimator"] == "hedges_g_exact"


def test_leave_one_out_ignores_previously_excluded_rows_and_keeps_global_ids() -> None:
    result = ma.meta_analysis(
        effect=[0.0, np.nan, 1.0, 2.0],
        variance=[0.04, 0.05, 0.04, 0.04],
        study=["A", "Missing", "B", "C"],
        model="common",
        missing="drop",
    )
    influence = result.leave_one_out()

    assert influence.table["omitted_row_id"].tolist() == [0, 2, 3]
    assert all(
        1 not in refit.study_results["row_id"].tolist() for refit in influence.results
    )
    assert all(refit.method.missing == "drop" for refit in influence.results)


def test_leave_one_out_study_count_boundaries_are_explicit() -> None:
    common = ma.meta_analysis(effect=[0.1, 0.2], variance=[0.01, 0.02], model="common")
    assert len(common.leave_one_out()) == 2
    assert all(refit.k == 1 for refit in common.leave_one_out().results)

    random = ma.meta_analysis(effect=[0.1, 0.2], variance=[0.01, 0.02], model="random")
    with pytest.raises(ma.InsufficientStudiesError, match="three included studies"):
        random.leave_one_out()


def _ordered_dataframe_result() -> ma.MetaAnalysisResult:
    data = pd.DataFrame(
        {
            "yi": [0.0, 0.2, 0.8, 1.0],
            "vi": [0.04, 0.04, 0.04, 0.04],
            "year": [2003, 2001, 2004, 2002],
        },
        index=["S1", "S2", "S3", "S4"],
    )
    return ma.meta_analysis(
        data,
        effect="yi",
        variance="vi",
        model="common",
    )


def test_cumulative_uses_dataframe_order_column_and_stable_row_ids() -> None:
    result = _ordered_dataframe_result()
    cumulative = result.cumulative(order="year")
    table = cumulative.to_dataframe()

    assert isinstance(cumulative, ma.CumulativeMetaAnalysisResult)
    assert table["added_row_ids"].tolist() == [(1,), (3,), (0,), (2,)]
    assert table["added_studies"].tolist() == [
        ("S2",),
        ("S4",),
        ("S1",),
        ("S3",),
    ]
    assert table["order_value"].tolist() == [2001, 2002, 2003, 2004]
    np.testing.assert_allclose(table["estimate"], [0.2, 0.6, 0.4, 0.5])
    assert cumulative.final.estimate == pytest.approx(result.estimate)
    assert cumulative.final.q == pytest.approx(result.q)


def test_cumulative_defaults_to_input_order_and_can_sort_descending() -> None:
    result = _ordered_dataframe_result()
    natural = result.cumulative()
    descending = result.cumulative(order="year", ascending=False)

    assert natural.table["added_row_ids"].tolist() == [(0,), (1,), (2,), (3,)]
    assert descending.table["order_value"].tolist() == [2004, 2003, 2002, 2001]
    assert descending.table["added_row_ids"].tolist() == [(2,), (0,), (3,), (1,)]


def test_cumulative_collapse_adds_equal_order_values_together() -> None:
    result = ma.meta_analysis(
        effect=[0.0, 0.2, 0.8, 1.0],
        variance=[0.04] * 4,
        study=["A", "B", "C", "D"],
        model="common",
    )
    cumulative = result.cumulative(
        order=[2000, 2000, 2001, 2002],
        collapse=True,
    )

    assert cumulative.table["k"].tolist() == [2, 3, 4]
    assert cumulative.table["added_row_ids"].tolist() == [(0, 1), (2,), (3,)]
    assert cumulative.table["order_value"].tolist() == [2000, 2001, 2002]


def test_random_cumulative_starts_at_first_estimable_prefix() -> None:
    result = ma.meta_analysis(
        effect=[0.0, 0.3, 0.9, 1.2],
        variance=[0.04] * 4,
        study=["A", "B", "C", "D"],
        model="random",
        tau2_method="PM",
    )
    cumulative = result.cumulative()

    assert cumulative.table["k"].tolist() == [2, 3, 4]
    assert cumulative.table.loc[0, "added_row_ids"] == (0, 1)
    assert "begins at k=2" in cumulative.warnings[0]
    assert all(refit.model == "random" for refit in cumulative.results)
    assert all(refit.method.tau2_method == "PM" for refit in cumulative.results)


def test_excluded_order_values_are_filtered_before_missing_validation() -> None:
    result = ma.meta_analysis(
        effect=[0.0, np.nan, 1.0],
        variance=[0.04, 0.04, 0.04],
        model="common",
        missing="drop",
    )
    cumulative = result.cumulative(order=[2, np.nan, 1])

    assert cumulative.table["added_row_ids"].tolist() == [(2,), (0,)]
    assert cumulative.table["k"].tolist() == [1, 2]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"order": [1, 2]}, "length 2"),
        ({"order": [[1, 2, 3, 4]]}, "one-dimensional"),
        ({"order": [1, np.nan, 3, 4]}, "must not be missing"),
        ({"order": [1, "two", 3, 4]}, "mutually sortable"),
        ({"collapse": True}, "requires an explicit order"),
        ({"ascending": "yes"}, "ascending must be a boolean"),
        ({"collapse": 1}, "collapse must be a boolean"),
        ({"order": "unknown"}, "Order column 'unknown'"),
    ],
)
def test_invalid_cumulative_controls_raise_domain_errors(
    kwargs: dict[str, object], match: str
) -> None:
    result = _ordered_dataframe_result()
    with pytest.raises(ma.InvalidStudyDataError, match=match):
        result.cumulative(**kwargs)  # type: ignore[arg-type]


def test_cumulative_final_fit_is_order_invariant() -> None:
    result = ma.meta_analysis(
        effect=[-0.2, 0.1, 0.6, 1.0],
        variance=[0.03, 0.04, 0.05, 0.06],
        model="common",
    )
    for order in itertools.permutations(range(4)):
        final = result.cumulative(order=order).final
        assert final.estimate == pytest.approx(result.estimate, abs=1e-14)
        assert final.q == pytest.approx(result.q, abs=1e-13)


def test_source_dataframe_is_retained_defensively() -> None:
    data = pd.DataFrame({"yi": [0.1, 0.2], "vi": [0.01, 0.02], "year": [2000, 2001]})
    result = ma.meta_analysis(data, effect="yi", variance="vi", model="common")
    data.loc[0, "year"] = 9999
    first = result.source_data
    assert first is not None
    assert first.loc[0, "year"] == 2000
    first.loc[0, "year"] = 8888
    second = result.source_data
    assert second is not None
    assert second.loc[0, "year"] == 2000


def _subgroup_result() -> ma.SubgroupMetaAnalysisResult:
    data = pd.DataFrame(
        {
            "yi": [0.0, 0.2, 0.4, 0.8, 1.0, 1.2],
            "vi": [0.04] * 6,
            "region": ["A", "A", "A", "B", "B", "B"],
            "year": [2003, 2001, 2002, 2003, 2001, 2002],
        },
        index=["A3", "A1", "A2", "B3", "B1", "B2"],
    )
    result = ma.meta_analysis(
        data,
        effect="yi",
        variance="vi",
        subgroup="region",
        model="common",
    )
    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    return result


def test_subgroup_leave_one_out_returns_group_and_overall_results() -> None:
    result = _subgroup_result()
    influence = result.leave_one_out()
    combined = influence.to_dataframe()

    assert isinstance(influence, ma.SubgroupLeaveOneOutResult)
    assert list(influence.groups) == ["A", "B"]
    assert len(influence.groups["A"]) == 3
    assert len(influence.groups["B"]) == 3
    assert len(influence.overall) == 6
    assert set(combined["scope"]) == {"subgroup", "overall"}
    with pytest.raises(TypeError):
        influence.groups["C"] = influence.groups["A"]  # type: ignore[index]


def test_subgroup_cumulative_resolves_source_column_within_each_group() -> None:
    result = _subgroup_result()
    cumulative = result.cumulative(order="year")

    assert isinstance(cumulative, ma.SubgroupCumulativeMetaAnalysisResult)
    assert cumulative.groups["A"].table["added_studies"].tolist() == [
        ("A1",),
        ("A2",),
        ("A3",),
    ]
    assert cumulative.groups["B"].table["added_studies"].tolist() == [
        ("B1",),
        ("B2",),
        ("B3",),
    ]
    assert cumulative.overall.table["order_value"].tolist() == [
        2001,
        2001,
        2002,
        2002,
        2003,
        2003,
    ]


def test_subgroup_cumulative_accepts_full_length_array_order() -> None:
    result = _subgroup_result()
    cumulative = result.cumulative(order=[3, 1, 2, 6, 4, 5])

    assert cumulative.groups["A"].table["added_row_ids"].tolist() == [
        (1,),
        (2,),
        (0,),
    ]
    assert cumulative.groups["B"].table["added_row_ids"].tolist() == [
        (4,),
        (5,),
        (3,),
    ]


def test_subgroup_leave_one_out_reports_small_group_context() -> None:
    result = ma.meta_analysis(
        effect=[0.0, 0.8, 1.0],
        variance=[0.04] * 3,
        subgroup=["A", "B", "B"],
        model="common",
    )
    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    with pytest.raises(ma.InsufficientStudiesError, match="Subgroup 'A'"):
        result.leave_one_out()


def test_sensitivity_tables_are_defensive_copies() -> None:
    result = _ordered_dataframe_result()
    cumulative = result.cumulative(order="year")
    influence = result.leave_one_out()

    cumulative_table = cumulative.table
    cumulative_table.loc[0, "estimate"] = 999.0
    assert cumulative.table.loc[0, "estimate"] != 999.0

    influence_table = influence.table
    influence_table.loc[0, "estimate"] = 999.0
    assert influence.table.loc[0, "estimate"] != 999.0
    pd.testing.assert_frame_equal(cumulative.summary(), cumulative.to_dataframe())
    pd.testing.assert_frame_equal(influence.summary(), influence.to_dataframe())
