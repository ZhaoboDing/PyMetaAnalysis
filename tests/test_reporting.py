"""Result provenance and reporting API tests."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma


def _generic_with_exclusion() -> ma.MetaAnalysisResult:
    data = pd.DataFrame(
        {
            "yi": [0.1, np.nan, 0.4],
            "vi": [0.01, 0.02, 0.03],
        },
        index=pd.Index(["A", "B", "C"], name="trial"),
    )
    return ma.meta_analysis(
        data,
        effect="yi",
        variance="vi",
        model="common",
        missing="drop",
    )


def _sparse_binary() -> ma.MetaAnalysisResult:
    return ma.meta_binary(
        event_treat=[0, 8, 0, 12],
        n_treat=[50, 60, 40, 70],
        event_control=[4, 6, 0, 10],
        n_control=[50, 60, 40, 70],
        measure="OR",
        method="MH",
    )


def test_dataframe_provenance_records_version_columns_and_row_decisions() -> None:
    result = _generic_with_exclusion()
    provenance = result.provenance

    assert isinstance(provenance, ma.AnalysisProvenance)
    assert provenance.package_version == ma.__version__
    assert provenance.schema_version == "1.0"
    assert provenance.analysis_type == "generic"
    assert provenance.data_source == "pandas_dataframe"
    assert dict(provenance.column_mapping) == {
        "effect": "yi",
        "variance": "vi",
    }
    assert provenance.input_fields[-1] == ma.InputFieldProvenance(
        role="study",
        source="dataframe_index",
    )
    assert provenance.row_count == 3
    assert provenance.included_rows == (0, 2)
    assert provenance.excluded_rows == (1,)

    with pytest.raises(TypeError):
        provenance.column_mapping["effect"] = "other"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        provenance.row_count = 10  # type: ignore[misc]


def test_array_inputs_and_generated_study_labels_are_explicit() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.2],
        variance=[0.01, 0.02],
        model="common",
    )

    fields = {field.role: field for field in result.provenance.input_fields}
    assert result.provenance.data_source == "array_like"
    assert fields["effect"].source == "array"
    assert fields["variance"].source == "array"
    assert fields["study"].source == "generated_row_number"
    assert dict(result.provenance.column_mapping) == {}


def test_binary_provenance_records_corrections_and_uninformative_rows() -> None:
    result = _sparse_binary()
    transformations = {
        record.name: record for record in result.provenance.transformations
    }

    assert result.provenance.analysis_type == "binary"
    assert transformations["binary_effect_size"].affected_rows == (0, 1, 3)
    assert dict(transformations["binary_effect_size"].parameters) == {
        "measure": "OR",
        "model_scale": "log",
        "display_scale": "exp",
    }
    assert transformations["continuity_correction"].affected_rows == (0,)
    assert dict(transformations["continuity_correction"].parameters) == {
        "value": 0.5,
        "scope": "only_zero_studies",
        "target": "individual_effects",
    }
    assert transformations["relative_effect_exclusion"].affected_rows == (2,)
    assert transformations["mantel_haenszel_continuity_correction"].affected_rows == ()
    assert result.provenance.excluded_rows == (2,)


def test_explicit_mh_correction_records_only_the_rows_it_changes() -> None:
    result = ma.meta_binary(
        event_treat=[0, 8, 12],
        n_treat=[50, 60, 70],
        event_control=[4, 6, 10],
        n_control=[50, 60, 70],
        measure="RR",
        method="MH",
        mh_continuity_correction=0.5,
    )
    transformations = {
        record.name: record for record in result.provenance.transformations
    }

    correction = transformations["mantel_haenszel_continuity_correction"]
    assert correction.affected_rows == (0,)
    assert dict(correction.parameters) == {
        "value": 0.5,
        "scope": "only_zero_studies",
        "target": "pooling",
    }


def test_continuous_provenance_records_the_resolved_smd_estimator() -> None:
    result = ma.meta_continuous(
        mean_treat=[4.0, 5.0],
        sd_treat=[2.0, 3.0],
        n_treat=[20, 25],
        mean_control=[3.0, 4.0],
        sd_control=[2.5, 2.0],
        n_control=[22, 24],
        measure="SMD",
        model="common",
    )

    [transformation] = result.provenance.transformations
    assert result.provenance.analysis_type == "continuous"
    assert transformation.name == "continuous_effect_size"
    assert dict(transformation.parameters) == {
        "measure": "SMD",
        "effect_estimator": "hedges_g_exact",
        "standardizer": "pooled_sd",
        "smd_variance": "LS",
    }
    assert transformation.affected_rows == (0, 1)
    assert "exact Hedges small-sample correction" in result.method_details()


def test_mean_difference_method_details_record_unpooled_variance() -> None:
    result = ma.meta_continuous(
        mean_treat=[4.0, 5.0],
        sd_treat=[2.0, 3.0],
        n_treat=[20, 25],
        mean_control=[3.0, 4.0],
        sd_control=[2.5, 2.0],
        n_control=[22, 24],
        measure="MD",
        model="common",
    )

    assert "Mean differences used unpooled sampling variances" in (
        result.method_details()
    )


def test_method_details_expand_random_effects_and_hartung_knapp_choices() -> None:
    result = ma.meta_analysis(
        effect=[-0.4, 0.1, 0.5, 1.2, 1.8],
        variance=[0.04, 0.09, 0.05, 0.16, 0.08],
        model="random",
        tau2_method="REML",
        ci_method="hartung_knapp_adhoc",
    )

    details = result.method_details()

    assert "random-effects meta-analysis of 5 studies" in details
    assert "Between-study variance was estimated with REML" in details
    assert "absolute tolerance 1e-10" in details
    assert "ad hoc lower-bound variance safeguard" in details
    assert "Higgins–Thompson–Spiegelhalter prediction interval" in details
    assert f"PyMetaAnalysis {ma.__version__}" in details


def test_random_effects_summary_mentions_hartung_knapp_without_a_warning() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.2, 0.4],
        variance=[0.01, 0.02, 0.03],
        model="random",
    )

    assert "consider ci_method='hartung_knapp'" in str(result.summary())
    assert not any("hartung" in warning.lower() for warning in result.warnings)


def test_binary_method_details_distinguish_effect_and_mh_corrections() -> None:
    details = _sparse_binary().method_details()

    assert "log odds ratios" in details
    assert "Mantel–Haenszel estimator" in details
    assert "Individual-study effects used continuity correction 0.5" in details
    assert "it affected 1 row(s)" in details
    assert "Mantel–Haenszel pooling used continuity correction 0" in details


def test_report_payload_contains_resolved_methods_diagnostics_and_studies() -> None:
    result = _generic_with_exclusion()
    report = result.report()
    payload = report.to_dict()

    assert isinstance(report, ma.ResultReport)
    assert payload["schema_version"] == "1.0"
    assert payload["report_type"] == "meta_analysis"
    assert payload["analysis"] == {
        "model": "common",
        "measure": "GENERIC",
        "effect_scale": "identity",
        "display_scale": "identity",
        "included_studies": 2,
        "total_rows": 3,
    }
    assert payload["method"]["pooling_method"] == "inverse_variance"
    assert payload["method"]["confidence_level"] == 0.95
    assert payload["diagnostics"]["converged"] is True
    assert payload["provenance"]["excluded_rows"] == [1]
    assert len(payload["studies"]) == 3
    assert payload["method_details"] == result.method_details()


def test_report_json_is_strict_and_converts_unavailable_values_to_null() -> None:
    excluded = json.loads(_generic_with_exclusion().report().to_json())["studies"][1]
    single = ma.meta_analysis(effect=[0.5], variance=[0.04], model="common")
    heterogeneity = json.loads(single.report().to_json(indent=None))["heterogeneity"]

    assert excluded["effect"] is None
    assert excluded["weight"] is None
    assert excluded["normalized_weight"] is None
    assert heterogeneity["pvalue"] is None
    assert heterogeneity["i2"] is None
    assert heterogeneity["h2"] is None
    assert "p=not available" in single.report().to_markdown()


def test_report_is_detached_and_studies_can_be_omitted() -> None:
    report = _generic_with_exclusion().report(include_studies=False)
    first = report.to_dict()
    first["analysis"]["model"] = "changed"
    second = report.to_dict()

    assert "studies" not in first
    assert second["analysis"]["model"] == "common"


def test_markdown_report_separates_display_and_model_scales() -> None:
    result = ma.meta_binary(
        event_treat=[12, 5, 20, 7],
        n_treat=[100, 80, 120, 90],
        event_control=[18, 9, 15, 10],
        n_control=[110, 75, 130, 95],
        measure="RR",
        method="IV",
        model="random",
    )

    markdown = result.report(include_studies=False).to_markdown()

    assert markdown.startswith("# Meta-analysis report")
    assert "## Results" in markdown
    assert "## Methods" in markdown
    assert "## Provenance and diagnostics" in markdown
    assert "Model-scale estimate:" in markdown
    assert "Prediction interval:" in markdown
    assert str(result.report(include_studies=False)) == markdown


def test_report_serializes_timestamp_study_labels() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.2],
        variance=[0.01, 0.02],
        study=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
        model="common",
    )

    studies = json.loads(result.report().to_json())["studies"]

    assert studies[0]["study"] == "2024-01-01T00:00:00"
    assert studies[1]["study"] == "2024-02-01T00:00:00"


def test_subgroup_report_records_mapping_groups_and_test_assumptions() -> None:
    data = pd.DataFrame(
        {
            "yi": [0.1, 0.2, 0.8, 0.9],
            "vi": [0.02, 0.02, 0.02, 0.02],
            "region": ["A", "A", "B", "B"],
        }
    )
    result = ma.meta_analysis(
        data,
        effect="yi",
        variance="vi",
        subgroup="region",
        model="common",
    )
    payload = json.loads(result.report().to_json())

    assert isinstance(result, ma.SubgroupMetaAnalysisResult)
    assert dict(result.overall.provenance.column_mapping) == {
        "effect": "yi",
        "variance": "vi",
        "subgroup": "region",
    }
    assert [group.provenance.included_rows for group in result.groups.values()] == [
        (0, 1),
        (2, 3),
    ]
    assert payload["report_type"] == "subgroup_meta_analysis"
    assert [group["label"] for group in payload["groups"]] == ["A", "B"]
    assert payload["subgroup_differences"]["test_method"] == (
        "fixed_effect_on_subgroup_estimates"
    )
    assert len(payload["studies"]) == 4
    assert "Subgroup effects were fitted separately" in result.method_details()
    assert "Test for subgroup differences" in result.report().to_markdown()


def test_sensitivity_refits_remap_provenance_to_original_row_ids() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.2, 0.3, 0.4],
        variance=[0.01, 0.02, 0.03, 0.04],
        model="common",
    )

    first_refit = result.leave_one_out().results[0]

    assert first_refit.study_results["row_id"].tolist() == [1, 2, 3]
    assert first_refit.provenance.data_source == "derived_subset"
    assert first_refit.provenance.included_rows == (1, 2, 3)
    assert first_refit.provenance.excluded_rows == ()


def test_binary_sensitivity_refits_remap_transformation_rows() -> None:
    first_refit = _sparse_binary().leave_one_out().results[0]
    transformations = {
        record.name: record for record in first_refit.provenance.transformations
    }

    assert first_refit.study_results["row_id"].tolist() == [1, 3]
    assert first_refit.provenance.included_rows == (1, 3)
    assert transformations["binary_effect_size"].affected_rows == (1, 3)
