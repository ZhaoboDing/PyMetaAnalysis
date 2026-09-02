"""Meta-analysis of correlations on Fisher's z scale."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
DATA = pd.read_csv(REFERENCE_DIR / "correlation_input.csv")
REFERENCE: dict[str, Any] = json.loads(
    (REFERENCE_DIR / "correlation_metafor.json").read_text(encoding="utf-8")
)


def test_common_effect_matches_metafor_and_uses_display_scale() -> None:
    result = ma.meta_correlation(
        DATA,
        correlation="correlation",
        n="n",
        study="study",
        model="common",
    )
    expected = REFERENCE["common"]

    np.testing.assert_allclose(
        result.study_results["effect"],
        REFERENCE["effect"],
        rtol=5e-15,
        atol=5e-16,
    )
    np.testing.assert_allclose(
        result.study_results["variance"],
        REFERENCE["variance"],
        rtol=5e-15,
        atol=5e-16,
    )
    np.testing.assert_allclose(
        [result.estimate, result.standard_error, *result.ci],
        [expected["estimate"], expected["standard_error"], *expected["ci"]],
        rtol=5e-15,
        atol=5e-16,
    )
    np.testing.assert_allclose(
        [result.display_estimate, *result.display_ci],
        [expected["display_estimate"], *expected["display_ci"]],
        rtol=5e-15,
        atol=5e-16,
    )
    assert result.measure == "ZCOR"
    assert result.effect_scale == "fisher_z"
    assert result.display_scale == "tanh"
    assert result.study_results["effect_display"].tolist() == pytest.approx(
        DATA["correlation"].tolist()
    )
    assert result.study_results["normalized_weight"].sum() == pytest.approx(1.0)


def test_default_random_effects_matches_metafor_reml() -> None:
    data = DATA.drop(columns="study").copy()
    data.index = pd.Index([f"Study {index}" for index in range(len(data))])
    result = ma.meta_correlation(data, correlation="correlation", n="n")
    expected = REFERENCE["random_reml"]

    np.testing.assert_allclose(
        [result.estimate, result.standard_error, *result.ci, result.tau2],
        [
            expected["estimate"],
            expected["standard_error"],
            *expected["ci"],
            expected["tau2"],
        ],
        rtol=2e-10,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        result.study_results["normalized_weight"],
        expected["weights"],
        rtol=2e-10,
        atol=2e-11,
    )
    assert result.study_results["study"].tolist() == data.index.tolist()
    assert result.model == "random"
    assert result.method.tau2_method == "REML"
    assert result.prediction_interval is not None


def test_method_and_provenance_record_fisher_z_choices() -> None:
    result = ma.meta_correlation(
        DATA,
        correlation="correlation",
        n="n",
        study="study",
        model="common",
    )

    assert dict(result.method.options) == {
        "effect_transformation": "fisher_r_to_z",
        "sampling_variance": "1/(n-3)",
        "display_transformation": "tanh",
    }
    assert result.provenance.analysis_type == "correlation"
    assert result.provenance.column_mapping == {
        "correlation": "correlation",
        "n": "n",
        "study": "study",
    }
    [transformation] = result.provenance.transformations
    assert transformation.name == "fisher_r_to_z"
    assert transformation.affected_rows == tuple(range(len(DATA)))
    assert dict(transformation.parameters) == {
        "measure": "ZCOR",
        "sampling_variance": "1/(n-3)",
        "display_transformation": "tanh",
    }
    details = result.method_details()
    assert "Fisher's z-transformed correlations" in details
    assert "sampling variance 1 / (n - 3)" in details
    report = result.report().to_dict()
    assert report["analysis"]["display_scale"] == "tanh"
    assert report["results"]["display_estimate"] == pytest.approx(
        result.display_estimate
    )


def test_missing_rows_are_retained_with_specific_reasons() -> None:
    result = ma.meta_correlation(
        correlation=[0.2, np.nan, 0.4, np.nan, 0.5],
        n=[20, 30, np.nan, np.nan, 40],
        model="common",
        missing="drop",
    )
    studies = result.study_results

    assert result.k == 2
    assert studies["included"].tolist() == [True, False, False, False, True]
    assert studies["exclusion_reason"].tolist() == [
        None,
        "missing correlation",
        "missing n",
        "missing correlation, n",
        None,
    ]
    assert studies.loc[~studies["included"], "weight"].isna().all()
    assert any("Excluded 3 study" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ({"correlation": [-1.0]}, "strictly between -1 and 1"),
        ({"correlation": [1.0]}, "strictly between -1 and 1"),
        ({"correlation": [np.inf]}, "correlation must be finite"),
        ({"n": [3]}, "at least 4"),
        ({"n": [4.5]}, "whole-number"),
        ({"n": [np.inf]}, "n must be finite"),
        ({"correlation": ["large"]}, "numeric values"),
    ],
)
def test_invalid_inputs_raise_domain_errors(
    replacement: dict[str, list[object]],
    match: str,
) -> None:
    inputs: dict[str, list[object]] = {"correlation": [0.2], "n": [20]}
    inputs.update(replacement)

    with pytest.raises(ma.InvalidStudyDataError, match=match):
        ma.meta_correlation(**inputs, model="common")  # type: ignore[arg-type]


def test_unsupported_raw_correlation_pooling_is_explicit() -> None:
    with pytest.raises(ma.UnsupportedMethodError, match="only 'ZCOR'"):
        ma.meta_correlation(
            correlation=[0.2, 0.4],
            n=[20, 30],
            measure="COR",
            model="common",
        )


def test_input_shape_empty_and_missing_policy_errors_are_specific() -> None:
    with pytest.raises(ma.InvalidStudyDataError, match="same length"):
        ma.meta_correlation(correlation=[0.2, 0.3], n=[20], model="common")
    with pytest.raises(ma.InvalidStudyDataError, match="At least one study row"):
        ma.meta_correlation(correlation=[], n=[], model="common")
    with pytest.raises(ma.InvalidStudyDataError, match="one value per DataFrame row"):
        ma.meta_correlation(
            pd.DataFrame({"unused": [1, 2, 3]}),
            correlation=[0.2, 0.3],
            n=[20, 30],
            model="common",
        )
    with pytest.raises(ma.InvalidStudyDataError, match="Missing correlation inputs"):
        ma.meta_correlation(correlation=[np.nan], n=[20], model="common")
    with pytest.raises(ma.InvalidStudyDataError, match="No studies remain"):
        ma.meta_correlation(
            correlation=[np.nan],
            n=[20],
            model="common",
            missing="drop",
        )
    with pytest.raises(ma.InvalidStudyDataError, match="missing must be"):
        ma.meta_correlation(
            correlation=[0.2],
            n=[20],
            model="common",
            missing="omit",  # type: ignore[arg-type]
        )


def test_sign_symmetry_and_row_order_invariance() -> None:
    correlation = DATA["correlation"].to_numpy()
    n = DATA["n"].to_numpy()
    forward = ma.meta_correlation(correlation=correlation, n=n, model="common")
    reversed_sign = ma.meta_correlation(
        correlation=-correlation,
        n=n,
        model="common",
    )
    order = np.arange(len(correlation))[::-1]
    reordered = ma.meta_correlation(
        correlation=correlation[order],
        n=n[order],
        model="common",
    )

    assert reversed_sign.estimate == pytest.approx(-forward.estimate)
    assert reversed_sign.ci == pytest.approx((-forward.ci_high, -forward.ci_low))
    assert reversed_sign.display_estimate == pytest.approx(-forward.display_estimate)
    assert reversed_sign.display_ci == pytest.approx(
        (-forward.display_ci[1], -forward.display_ci[0])
    )
    assert reordered.estimate == pytest.approx(forward.estimate)
    assert reordered.standard_error == pytest.approx(forward.standard_error)
    assert reordered.q == pytest.approx(forward.q)


def test_subgroup_and_sensitivity_workflows_reconstruct_correlation_inputs() -> None:
    data = DATA.assign(subgroup=["early"] * 4 + ["late"] * 4)
    subgroup = ma.meta_correlation(
        data,
        correlation="correlation",
        n="n",
        study="study",
        subgroup="subgroup",
        model="common",
    )

    assert isinstance(subgroup, ma.SubgroupMetaAnalysisResult)
    assert list(subgroup.groups) == ["early", "late"]
    assert subgroup.overall.provenance.column_mapping["subgroup"] == "subgroup"
    assert np.isfinite(subgroup.q_between)

    result = subgroup.overall
    leave_one_out = result.leave_one_out()
    direct = ma.meta_correlation(
        data.iloc[1:],
        correlation="correlation",
        n="n",
        study="study",
        model="common",
    )
    assert leave_one_out.results[0] is not None
    assert leave_one_out.results[0].estimate == pytest.approx(direct.estimate)
    assert leave_one_out.results[0].study_results["row_id"].tolist() == list(
        range(1, len(data))
    )

    cumulative = result.cumulative()
    assert cumulative.final.estimate == pytest.approx(result.estimate)
    assert cumulative.final.display_estimate == pytest.approx(result.display_estimate)


def test_duplicate_study_labels_are_reported() -> None:
    result = ma.meta_correlation(
        correlation=[0.2, 0.3],
        n=[20, 30],
        study=["same", "same"],
        model="common",
    )

    assert any("Duplicate study labels" in warning for warning in result.warnings)
