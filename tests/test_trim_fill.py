from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from meta_analyze import (
    ConvergenceError,
    InsufficientStudiesError,
    UnsupportedMethodError,
    meta_analysis,
    meta_binary,
)

EFFECT = np.array([0.10, 0.15, 0.18, 0.22, 0.24, 0.29, 0.33, 0.37, 0.95, 1.15])
VARIANCE = np.array([0.04, 0.035, 0.03, 0.028, 0.025, 0.022, 0.02, 0.018, 0.012, 0.01])


REFERENCE = json.loads(
    (Path(__file__).parent / "reference" / "trimfill_metafor.json").read_text()
)


@pytest.mark.parametrize("key", REFERENCE["cases"])
def test_trim_fill_matches_metafor(key: str) -> None:
    method, estimator, side = key.split("_")
    model = "common" if method == "fe" else "random"
    expected = REFERENCE["cases"][key]
    result = meta_analysis(effect=EFFECT, variance=VARIANCE, model=model)
    filled = result.trim_and_fill(side=side, estimator=estimator)  # type: ignore[arg-type]
    assert filled.k0 == expected["k0"]
    assert filled.adjusted_estimate == pytest.approx(
        expected["estimate"][0][0], abs=1e-6
    )
    assert filled.adjusted_tau2 == pytest.approx(expected["tau2"], abs=1e-6)
    assert filled.k0_standard_error == pytest.approx(
        expected["k0_standard_error"], abs=1e-6
    )
    assert filled.k0_pvalue == expected["k0_pvalue"]


def test_automatic_side_matches_metafor_regression_direction() -> None:
    result = meta_analysis(effect=EFFECT, variance=VARIANCE, model="random")
    automatic = result.trim_and_fill()
    explicit = result.trim_and_fill(side="right")
    assert automatic.side == "right"
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
    assert "sensitivity analysis" in str(filled)


def test_rejects_invalid_controls_and_unsupported_pooling() -> None:
    result = meta_analysis(effect=EFFECT, variance=VARIANCE)
    with pytest.raises(UnsupportedMethodError):
        result.trim_and_fill(side="up")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedMethodError):
        result.trim_and_fill(estimator="Q0")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        result.trim_and_fill(estimator=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        result.trim_and_fill(max_iterations=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
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
    result = meta_analysis(effect=EFFECT, variance=VARIANCE, model="common")
    with pytest.raises(ConvergenceError):
        result.trim_and_fill(side="right", max_iterations=1)
