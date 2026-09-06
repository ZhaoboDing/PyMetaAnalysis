from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meta_analyze import (
    ConvergenceError,
    InsufficientStudiesError,
    InvalidStudyDataError,
    UnsupportedMethodError,
    meta_analysis,
    meta_binary,
    meta_correlation,
)

EFFECT = np.array([0.10, 0.15, 0.18, 0.22, 0.24, 0.29, 0.33, 0.37, 0.95, 1.15])
VARIANCE = np.array([0.04, 0.035, 0.03, 0.028, 0.025, 0.022, 0.02, 0.018, 0.012, 0.01])


REFERENCE = json.loads(
    (Path(__file__).parent / "reference" / "trimfill_metafor.json").read_text()
)


@pytest.mark.parametrize("dataset", ["ordinary", "tied"])
@pytest.mark.parametrize("key", REFERENCE["cases"])
def test_trim_fill_matches_metafor(key: str, dataset: str) -> None:
    method, estimator, side = key.split("_")
    model = "common" if method == "fe" else "random"
    data = REFERENCE if dataset == "ordinary" else REFERENCE["tied"]
    expected = data["cases"][key]
    result = meta_analysis(effect=data["yi"], variance=data["vi"], model=model)
    filled = result.trim_and_fill(
        side=None if side == "auto" else side, estimator=estimator
    )  # type: ignore[arg-type]
    assert filled.k0 == expected["k0"]
    assert filled.side == expected["side"]
    # Closed forms use tight rounding tolerances; REML and mirrored rows include
    # root-solver error at the explicitly recorded 1e-10 R/Python controls.
    rtol, atol = (5e-13, 5e-15) if method == "fe" else (2e-10, 2e-11)
    assert filled.adjusted_estimate == pytest.approx(
        expected["estimate"][0][0], rel=rtol, abs=atol
    )
    assert filled.adjusted_tau2 == pytest.approx(expected["tau2"], rel=rtol, abs=atol)
    if expected["k0_standard_error"] is None:
        assert np.isnan(filled.k0_standard_error)
    else:
        assert filled.k0_standard_error == pytest.approx(
            expected["k0_standard_error"], rel=5e-13, abs=5e-15
        )
    assert filled.k0_pvalue == expected["k0_pvalue"]
    adjusted = filled.adjusted_result
    for name in ("standard_error", "ci", "q", "i2", "h2"):
        np.testing.assert_allclose(
            getattr(adjusted, name), expected[name], rtol=rtol, atol=atol
        )
    for name in ("effect", "variance"):
        np.testing.assert_allclose(
            filled.augmented_studies[name], expected[name], rtol=rtol, atol=atol
        )
    assert filled.augmented_studies.imputed.tolist() == expected["imputed"]


def test_automatic_side_matches_metafor_regression_direction() -> None:
    result = meta_analysis(effect=EFFECT, variance=VARIANCE, model="random")
    automatic = result.trim_and_fill()
    explicit = result.trim_and_fill(side="right")
    assert automatic.side == "right"
    assert automatic.side_selection == "standard_error_meta_regression"
    assert automatic.side_selection_statistic is not None
    assert automatic.side_selection_statistic < 0
    assert automatic.iteration_trace[-1] == automatic.k0
    assert automatic.adjusted_estimate == pytest.approx(explicit.adjusted_estimate)


def test_augmented_table_is_defensive_and_marks_provenance() -> None:
    result = meta_analysis(effect=EFFECT, variance=VARIANCE, model="common")
    filled = result.trim_and_fill(side="right")
    table = filled.augmented_studies
    assert table["imputed"].sum() == filled.k0 == 3
    assert table.loc[table["imputed"], "mirror_source"].notna().all()
    table.loc[:, "effect"] = 999
    assert not (filled.augmented_studies["effect"] == 999).any()
    record = filled.adjusted_result.provenance.transformations[-1]
    assert record.name == "trim_and_fill_imputation"
    assert len(record.affected_rows) == 3
    assert filled.adjusted_result.measure == result.measure


def test_r0_probability_and_serialization() -> None:
    filled = meta_analysis(effect=EFFECT, variance=VARIANCE).trim_and_fill(
        side="left", estimator="R0"
    )
    payload = filled.to_dict()
    assert payload["k0_pvalue"] == 0.25
    assert len(payload["augmented_studies"]) == 11
    assert payload["original_ci"] == filled.original_result.ci
    assert payload["adjusted_ci"] == filled.adjusted_result.ci
    assert "sensitivity analysis" in str(filled)


def test_rejects_invalid_controls_and_unsupported_pooling() -> None:
    result = meta_analysis(effect=EFFECT, variance=VARIANCE)
    with pytest.raises(UnsupportedMethodError):
        result.trim_and_fill(side="up")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedMethodError):
        result.trim_and_fill(estimator="Q0")  # type: ignore[arg-type]
    with pytest.raises(InvalidStudyDataError):
        result.trim_and_fill(estimator=None)  # type: ignore[arg-type]
    with pytest.raises(InvalidStudyDataError):
        result.trim_and_fill(max_iterations=True)  # type: ignore[arg-type]
    with pytest.raises(InvalidStudyDataError):
        result.trim_and_fill(max_iterations=0)
    mh = meta_binary(
        event_treat=[1, 2],
        n_treat=[10, 10],
        event_control=[2, 3],
        n_control=[10, 10],
        method="mantel_haenszel",
        model="common",
    )
    with pytest.raises(UnsupportedMethodError):
        mh.trim_and_fill(side="left")


def test_study_count_and_convergence_errors() -> None:
    one = meta_analysis(effect=[0.2], variance=[0.1], model="common")
    with pytest.raises(InsufficientStudiesError):
        one.trim_and_fill(side="left")
    two = meta_analysis(effect=[0.1, 0.2], variance=[0.1, 0.1], model="common")
    with pytest.raises(InsufficientStudiesError):
        two.trim_and_fill(side="left")
    result = meta_analysis(effect=EFFECT, variance=VARIANCE, model="common")
    with pytest.raises(ConvergenceError):
        result.trim_and_fill(side="right", max_iterations=1)


def test_filled_funnel_marks_synthetic_studies_and_renders(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    filled = meta_analysis(
        effect=EFFECT, variance=VARIANCE, model="common"
    ).trim_and_fill(side="right")
    axes = filled.funnel(contour_levels=(0.90, 0.95), warn_on_few_studies=False)
    assert len(axes.collections) == 6
    assert axes.collections[-1].get_facecolors().size == 0
    assert {text.get_text() for text in axes.get_legend().get_texts()} >= {
        "Observed study",
        "Imputed study",
    }
    output = tmp_path / "trim-fill-funnel.png"
    axes.figure.savefig(output, dpi=100)
    assert output.stat().st_size > 1000
    plt.close(axes.figure)


def test_fill_labels_do_not_collide_and_source_is_unchanged() -> None:
    source = meta_analysis(
        effect=EFFECT,
        variance=VARIANCE,
        study=["Filled 1", *[f"Study {i}" for i in range(2, 11)]],
        model="common",
    )
    before = source.study_results
    filled = source.trim_and_fill(side="right")
    assert filled.augmented_studies["study"].astype(str).is_unique
    assert filled.source_result is source
    assert source.study_results.equals(before)


def test_k1000_performance_smoke() -> None:
    effect = np.linspace(-1.0, 1.0, 1000)
    variance = np.linspace(0.01, 0.1, 1000)
    result = meta_analysis(effect=effect, variance=variance, model="common")
    started = time.perf_counter()
    filled = result.trim_and_fill(side="left")
    assert time.perf_counter() - started < 5.0
    assert 0 <= filled.k0 < 1000


@pytest.mark.parametrize("side", ["left", "right"])
def test_excluded_rows_have_consistent_augmented_provenance(side: str) -> None:
    source = meta_analysis(
        effect=np.insert(EFFECT, 1, np.nan),
        standard_error=np.sqrt(np.insert(VARIANCE, 1, 0.1)),
        study=["duplicate"] * 11,
        model="common",
        missing="drop",
    )
    before = source.study_results
    filled = source.trim_and_fill(side=side)  # type: ignore[arg-type]
    assert filled.k0 == (0 if side == "left" else 3)
    table = filled.augmented_studies
    observed_ids = (0, *range(2, 11))
    assert tuple(table.loc[~table.imputed, "row_id"]) == observed_ids
    assert len(table) == 10 + filled.k0
    assert table.included.all()
    assert set(table.loc[table.imputed, "row_id"]).isdisjoint(before.row_id)
    assert set(table.loc[table.imputed, "mirror_source_row_id"]) <= set(observed_ids)
    provenance = filled.adjusted_result.provenance
    assert provenance.included_rows == tuple(table.row_id)
    assert provenance.excluded_rows == ()
    assert provenance.row_count == len(table)
    for record in provenance.transformations:
        assert set(record.affected_rows) <= set(table.row_id)
    pd.testing.assert_frame_equal(source.study_results, before)
    assert source.provenance.excluded_rows == (1,)
    assert filled.adjusted_result.report().to_dict()["provenance"][
        "included_rows"
    ] == list(table.row_id)


@pytest.mark.parametrize("estimator", ["L0", "R0"])
@pytest.mark.parametrize("model", ["common", "random"])
def test_identical_effects_do_not_invent_missing_studies(
    estimator: str, model: str
) -> None:
    source = meta_analysis(effect=[0.3] * 5, variance=[0.1] * 5, model=model)
    filled = source.trim_and_fill(estimator=estimator)  # type: ignore[arg-type]
    assert filled.k0 == 0
    assert filled.converged
    assert filled.iterations == 0
    assert filled.iteration_trace == ()
    assert filled.adjusted_estimate == source.estimate
    assert filled.adjusted_ci == source.ci
    assert filled.k0_pvalue is None
    assert np.isnan(filled.k0_standard_error)
    assert any("identical" in note for note in filled.warnings)


def test_random_trimming_reports_unestimable_refit_as_convergence_error() -> None:
    source = meta_correlation(correlation=[0.3, 0.5, 0.2], n=[40, 55, 30])
    with pytest.raises(ConvergenceError, match="trimming.*fewer than 2"):
        source.trim_and_fill()


def test_augmented_dataframe_is_excluded_from_repr_and_equality() -> None:
    filled = meta_analysis(
        effect=EFFECT, variance=VARIANCE, model="common"
    ).trim_and_fill(side="right")
    altered = replace(filled, _augmented_studies=pd.DataFrame({"private_value": [999]}))
    assert (filled == altered) is True
    assert "_augmented_studies=" not in repr(filled)
    assert "private_value" not in repr(altered)
