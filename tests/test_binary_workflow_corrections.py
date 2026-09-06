"""Subset corrections are recomputed from retained raw tables, with global IDs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

import meta_analyze as ma

SCOPES = ["if_any_zero", "only_zero_studies", "all_studies"]
MODELS = [
    ("IV", "common", None),
    ("IV", "random", None),
    ("MH", "common", None),
    ("MH", "common", 0.5),
]


@pytest.fixture
def trials() -> pd.DataFrame:
    # Row 2 is the only informative zero-cell study. Excluded zero/missing
    # rows in group B must not trigger correction in that otherwise nonzero group.
    return pd.DataFrame(
        {
            "a": [4, 0, 0, 7, 10, np.nan, 5],
            "nt": [40, 45, 35, 60, 75, 50, 55],
            "c": [6, 0, 3, 9, 12, 4, 8],
            "nc": [45, 50, 40, 65, 80, 55, 60],
            "group": ["A", "B", "A", "B", "B", "B", "B"],
            "year": [2001, 2000, 2010, 2002, 2002, 2004, 2005],
        },
        index=["duplicate"] * 7,
    )


def _options(scope: str, method: str, model: str, mh_correction: float | None) -> dict:
    options = dict(
        event_treat="a",
        n_treat="nt",
        event_control="c",
        n_control="nc",
        measure="OR",
        method=method,
        model=model,
        missing="drop",
        correction_scope=scope,
    )
    if mh_correction is not None:
        options.update(
            mh_continuity_correction=mh_correction, mh_correction_scope=scope
        )
    return options


def _assert_raw_table_oracle(
    result: ma.MetaAnalysisResult,
    trials: pd.DataFrame,
    scope: str,
    mh_correction: float | None,
) -> None:
    table = result.study_results.set_index("row_id")
    ids = table.index[table.included].tolist()
    raw = trials.iloc[ids]
    cells = np.column_stack([raw.a, raw.nt - raw.a, raw.c, raw.nc - raw.c])
    zero = (cells == 0).any(axis=1)
    corrected = {
        "if_any_zero": np.full(len(ids), zero.any()),
        "only_zero_studies": zero,
        "all_studies": np.ones(len(ids), dtype=bool),
    }[scope]
    study_cells = cells + 0.5 * corrected[:, None]
    a, b, c, d = study_cells.T
    effect = np.log(a * d / (b * c))
    variance = (1 / study_cells).sum(axis=1)
    np.testing.assert_allclose(table.loc[ids, "effect"], effect, rtol=2e-13)
    np.testing.assert_allclose(table.loc[ids, "variance"], variance, rtol=2e-13)
    assert table.loc[ids, "continuity_corrected"].tolist() == corrected.tolist()
    assert not table.loc[~table.included, "continuity_corrected"].any()
    records = {record.name: record for record in result.provenance.transformations}
    corrected_ids = tuple(np.asarray(ids)[corrected])
    assert records["continuity_correction"].affected_rows == corrected_ids
    assert dict(records["continuity_correction"].parameters)["scope"] == scope
    assert result.provenance.included_rows == tuple(ids)
    assert result.provenance.excluded_rows == tuple(table.index[~table.included])
    assert dict(result.method.options)["correction_scope"] == scope

    if result.method.pooling_method == "inverse_variance":
        weights = 1 / (variance + result.tau2)
        estimate = np.dot(weights, effect) / weights.sum()
        assert result.standard_error**2 == pytest.approx(1 / weights.sum(), rel=2e-13)
    else:
        pooling_cells = cells.copy()
        if mh_correction is not None:
            pooling_cells += mh_correction * corrected[:, None]
        a, b, c, d = pooling_cells.T
        total = pooling_cells.sum(axis=1)
        estimate = np.log(np.sum(a * d / total) / np.sum(b * c / total))
        expected_mh = (
            corrected if mh_correction is not None else np.zeros(len(ids), bool)
        )
        assert (
            table.loc[ids, "mh_continuity_corrected"].tolist() == expected_mh.tolist()
        )
        assert records["mantel_haenszel_continuity_correction"].affected_rows == tuple(
            np.asarray(ids)[expected_mh]
        )
    assert result.estimate == pytest.approx(estimate, rel=2e-13, abs=2e-15)
    q_center = (
        np.sum(effect / variance) / np.sum(1 / variance)
        if result.method.pooling_method == "inverse_variance"
        else estimate
    )
    assert result.q == pytest.approx(
        np.sum((effect - q_center) ** 2 / variance), rel=2e-13
    )
    report = result.report()
    payload = json.loads(report.to_json())
    assert payload == report.to_dict()
    assert payload["provenance"]["included_rows"] == ids


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize(("method", "model", "mh_correction"), MODELS)
def test_subgroup_corrections_and_between_test(
    trials: pd.DataFrame,
    scope: str,
    method: str,
    model: str,
    mh_correction: float | None,
) -> None:
    grouped = ma.meta_binary(
        trials, subgroup="group", **_options(scope, method, model, mh_correction)
    )
    for result in [grouped.overall, *grouped.groups.values()]:
        _assert_raw_table_oracle(result, trials, scope, mh_correction)
    first, second = grouped.groups.values()
    expected_q = (first.estimate - second.estimate) ** 2 / (
        first.standard_error**2 + second.standard_error**2
    )
    assert grouped.q_between == pytest.approx(expected_q, rel=2e-13)
    assert grouped.q_between_pvalue == pytest.approx(chi2.sf(expected_q, 1), rel=2e-13)
    overall = grouped.overall.study_results.set_index("row_id")
    group_b = second.study_results.set_index("row_id")
    if scope == "if_any_zero":
        assert overall.loc[3, "continuity_corrected"]
        assert not group_b.loc[3, "continuity_corrected"]
        assert group_b.loc[3, "effect"] != overall.loc[3, "effect"]
    else:
        np.testing.assert_allclose(
            group_b.loc[[3, 4, 6], "effect"], overall.loc[[3, 4, 6], "effect"]
        )


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize(("method", "model", "mh_correction"), MODELS)
def test_leave_one_out_recomputes_correction_after_only_zero_study_is_removed(
    trials: pd.DataFrame,
    scope: str,
    method: str,
    model: str,
    mh_correction: float | None,
) -> None:
    source = ma.meta_binary(trials, **_options(scope, method, model, mh_correction))
    before = source.study_results
    workflow = source.leave_one_out()
    assert workflow.table.omitted_row_id.tolist() == [0, 2, 3, 4, 6]
    assert workflow.table.refit_success.all()
    for omitted, result in zip(
        workflow.table.omitted_row_id, workflow.results, strict=True
    ):
        assert result is not None
        _assert_raw_table_oracle(result, trials, scope, mh_correction)
        ids = set(result.provenance.included_rows)
        assert ids == {0, 2, 3, 4, 6} - {omitted}
    without_zero = workflow.results[1]
    assert without_zero is not None
    flags = without_zero.study_results.continuity_corrected
    assert flags.all() if scope == "all_studies" else not flags.any()
    pd.testing.assert_frame_equal(source.study_results, before)


@pytest.mark.parametrize("collapse", [False, True])
@pytest.mark.parametrize(("method", "model", "mh_correction"), MODELS)
def test_cumulative_addition_of_zero_study_revises_earlier_corrections(
    trials: pd.DataFrame,
    collapse: bool,
    method: str,
    model: str,
    mh_correction: float | None,
) -> None:
    source = ma.meta_binary(
        trials, **_options("if_any_zero", method, model, mh_correction)
    )
    workflow = source.cumulative(order="year", collapse=collapse)
    for result in workflow.results:
        _assert_raw_table_oracle(result, trials, "if_any_zero", mh_correction)
        table = result.study_results
        assert (
            table.continuity_corrected.all()
            if 2 in table.row_id.values
            else not table.continuity_corrected.any()
        )
    assert set(workflow.results[-2].provenance.included_rows) == {0, 3, 4, 6}
    assert set(workflow.final.provenance.included_rows) == {0, 2, 3, 4, 6}
    assert workflow.final.estimate == pytest.approx(source.estimate, rel=2e-13)
