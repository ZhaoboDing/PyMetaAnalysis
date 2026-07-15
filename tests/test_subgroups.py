from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import meta_analyze as ma


def _generic_subgroups() -> ma.SubgroupMetaAnalysisResult:
    result = ma.meta_analysis(
        effect=[0.0, 0.2, 0.8, 1.0],
        variance=[0.04, 0.04, 0.04, 0.04],
        subgroup=["A", "A", "B", "B"],
        model="common",
    )
    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    return result


def test_generic_subgroup_estimates_and_between_test_match_hand_calculation() -> None:
    result = _generic_subgroups()

    assert list(result.groups) == ["A", "B"]
    assert result.groups["A"].estimate == pytest.approx(0.1)
    assert result.groups["B"].estimate == pytest.approx(0.9)
    assert result.overall.estimate == pytest.approx(0.5)
    assert result.q_between == pytest.approx(16.0)
    assert result.q_between_df == 1
    assert result.q_between_pvalue == pytest.approx(chi2.sf(16.0, 1))
    assert result.i2_between == pytest.approx(15.0 / 16.0)
    assert result.method.test_method == "fixed_effect_on_subgroup_estimates"
    assert result.method.tau2_strategy == "not_applicable"
    assert all(
        group.study_results["normalized_weight"].sum() == pytest.approx(1.0)
        for group in result.groups.values()
    )


def test_subgroup_test_uses_pooled_standard_errors() -> None:
    result = _generic_subgroups()
    first, second = result.groups.values()
    expected = (first.estimate - second.estimate) ** 2 / (
        first.standard_error**2 + second.standard_error**2
    )

    assert result.q_between == pytest.approx(expected)


def test_dataframe_column_preserves_group_order_labels_and_global_row_ids() -> None:
    data = pd.DataFrame(
        {
            "yi": [0.8, 1.0, 0.0, 0.2],
            "vi": [0.04] * 4,
            "region": ["B", "B", "A", "A"],
        },
        index=["B1", "B2", "A1", "A2"],
    )
    result = ma.meta_analysis(
        data,
        effect="yi",
        variance="vi",
        subgroup="region",
        model="common",
    )

    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    assert list(result.groups) == ["B", "A"]
    assert result.study_results["study"].tolist() == list(data.index)
    assert result.study_results["subgroup"].tolist() == data["region"].tolist()
    assert result.groups["A"].study_results["row_id"].tolist() == [2, 3]


def test_binary_mantel_haenszel_supports_subgroups() -> None:
    result = ma.meta_binary(
        event_treat=[12, 5, 20, 7],
        n_treat=[100, 80, 120, 90],
        event_control=[18, 9, 15, 10],
        n_control=[110, 75, 130, 95],
        subgroup=["early", "early", "late", "late"],
        measure="RR",
        method="MH",
        model="common",
    )

    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    assert all(
        group.method.pooling_method == "mantel_haenszel"
        for group in result.groups.values()
    )
    assert result.overall.display_scale == "exp"
    assert np.isfinite(result.q_between)


def test_continuous_entry_supports_array_subgroups() -> None:
    result = ma.meta_continuous(
        mean_treat=[2.1, 2.4, 4.0, 4.4],
        sd_treat=[1.0, 1.1, 1.2, 1.1],
        n_treat=[20, 22, 24, 26],
        mean_control=[2.0, 2.0, 3.0, 3.1],
        sd_control=[1.0, 1.0, 1.1, 1.2],
        n_control=[20, 21, 23, 25],
        subgroup=["low", "low", "high", "high"],
        measure="MD",
        model="common",
    )

    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    assert result.groups["low"].measure == "MD"
    assert result.groups["high"].estimate > result.groups["low"].estimate


def test_random_subgroups_record_independent_tau2_strategy() -> None:
    result = ma.meta_analysis(
        effect=[0.0, 0.2, 0.6, 1.0, 1.4, 2.0],
        variance=[0.04] * 6,
        subgroup=["A", "A", "A", "B", "B", "B"],
        model="random",
        tau2_method="PM",
    )

    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    assert result.method.tau2_strategy == "independent"
    assert all(group.method.tau2_method == "PM" for group in result.groups.values())
    assert all(group.tau2 >= 0.0 for group in result.groups.values())


def test_outcome_exclusions_remain_visible_within_their_group() -> None:
    result = ma.meta_analysis(
        effect=[0.1, np.nan, 0.7, 0.9],
        variance=[0.02, 0.03, 0.04, 0.05],
        subgroup=["A", "A", "B", "B"],
        model="common",
        missing="drop",
    )

    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    excluded = result.excluded_studies
    assert excluded["row_id"].tolist() == [1]
    assert excluded["subgroup"].tolist() == ["A"]
    assert result.groups["A"].study_results["row_id"].tolist() == [0, 1]
    assert result.groups["A"].excluded_studies["row_id"].tolist() == [1]


@pytest.mark.parametrize(
    ("subgroup", "match"),
    [
        (["A", "A"], "at least two distinct"),
        (["A", None], "Missing subgroup labels"),
        (["A"], "length 1"),
    ],
)
def test_invalid_subgroup_inputs_raise_domain_errors(
    subgroup: list[object], match: str
) -> None:
    with pytest.raises(ma.MetaAnalysisError, match=match):
        ma.meta_analysis(
            effect=[0.1, 0.2],
            variance=[0.01, 0.02],
            subgroup=subgroup,
            model="common",
        )


def test_group_with_no_included_studies_is_not_silently_discarded() -> None:
    with pytest.raises(ma.InsufficientStudiesError, match="Subgroup 'A'"):
        ma.meta_analysis(
            effect=[np.nan, 0.2, 0.3],
            variance=[0.01, 0.02, 0.03],
            subgroup=["A", "B", "B"],
            model="common",
            missing="drop",
        )


def test_random_subgroup_requires_two_included_studies_per_group() -> None:
    with pytest.raises(ma.InsufficientStudiesError, match="Subgroup 'A'.*at least two"):
        ma.meta_analysis(
            effect=[0.1, 0.2, 0.3],
            variance=[0.01, 0.02, 0.03],
            subgroup=["A", "B", "B"],
            model="random",
        )


def test_result_mapping_and_tables_are_defensive() -> None:
    result = _generic_subgroups()
    with pytest.raises(TypeError):
        result.groups["C"] = result.groups["A"]  # type: ignore[index]

    table = result.study_results
    table.loc[0, "subgroup"] = "changed"
    assert result.study_results.loc[0, "subgroup"] == "A"

    rendered = str(result.summary())
    values = result.summary().to_dict()
    assert "Test for subgroup differences" in rendered
    assert values["q_between"] == pytest.approx(16.0)
    assert list(values["groups"]) == ["A", "B"]
    pd.testing.assert_frame_equal(result.to_dataframe(), result.study_results)


def test_subgroup_forest_renders_groups_studies_and_overall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_show(*args: object, **kwargs: object) -> None:
        raise AssertionError("subgroup forest() must not call show()")

    monkeypatch.setattr(plt, "show", fail_show)
    result = _generic_subgroups()
    axes = result.forest()
    output = tmp_path / "subgroup-forest.png"
    axes.figure.savefig(output, dpi=100)
    labels = [tick.get_text().strip() for tick in axes.get_yticklabels()]

    assert output.stat().st_size > 1000
    assert labels == [
        "A",
        "0",
        "1",
        "A subtotal",
        "B",
        "2",
        "3",
        "B subtotal",
        "Overall",
    ]
    assert "Test for subgroup differences" in axes.texts[-1].get_text()
    plt.close(axes.figure)


def test_ratio_subgroup_forest_uses_log_display_axis() -> None:
    result = ma.meta_binary(
        event_treat=[12, 5, 20, 7],
        n_treat=[100, 80, 120, 90],
        event_control=[18, 9, 15, 10],
        n_control=[110, 75, 130, 95],
        subgroup=["early", "early", "late", "late"],
        measure="OR",
        method="MH",
    )
    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    axes = result.forest()

    assert axes.get_xscale() == "log"
    plt.close(axes.figure)


def test_missing_plot_extra_has_actionable_subgroup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _generic_subgroups()
    real_import = builtins.__import__

    def reject_matplotlib(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "matplotlib.pyplot":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_matplotlib)
    with pytest.raises(ImportError, match=r"PyMetaAnalysis\[plot\]"):
        result.forest()
