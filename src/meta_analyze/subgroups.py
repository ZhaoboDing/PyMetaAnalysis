"""Construction and formal comparison of independent study subgroups."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2

from .config import SubgroupMethodConfig
from .data import ColumnOrArray, _resolve_vector
from .exceptions import InsufficientStudiesError, InvalidStudyDataError
from .provenance import remap_provenance_rows
from .results import MetaAnalysisResult, SubgroupMetaAnalysisResult

GroupFitter = Callable[[NDArray[np.int64]], MetaAnalysisResult]


def _subgroup_labels(
    subgroup: ColumnOrArray,
    *,
    data: pd.DataFrame | None,
    length: int,
) -> NDArray[np.object_]:
    labels = _resolve_vector(subgroup, data=data, name="subgroup")
    if len(labels) != length:
        raise InvalidStudyDataError(
            f"subgroup has length {len(labels)}, but the analysis has {length} rows."
        )
    missing = np.asarray(pd.isna(labels), dtype=np.bool_)
    if np.any(missing):
        rows = np.flatnonzero(missing).tolist()
        raise InvalidStudyDataError(
            f"Missing subgroup labels are not supported; invalid row positions: {rows}."
        )
    object_labels = np.asarray(labels, dtype=object)
    for position, label in enumerate(object_labels):
        try:
            hash(label)
        except TypeError as error:
            raise InvalidStudyDataError(
                f"subgroup labels must be hashable; invalid row position: {position}."
            ) from error
    return object_labels


def _restore_global_rows(
    result: MetaAnalysisResult,
    *,
    positions: NDArray[np.int64],
    label: Hashable,
    source_data: pd.DataFrame | None,
) -> MetaAnalysisResult:
    studies = result.study_results
    studies["row_id"] = positions
    studies.insert(2, "subgroup", [label for _ in range(len(studies))])
    group_source = (
        None if source_data is None else source_data.iloc[positions].copy(deep=True)
    )
    return replace(
        result,
        provenance=remap_provenance_rows(result.provenance, positions.tolist()),
        _study_results=studies,
        _source_data=group_source,
    )


def _between_group_test(
    groups: dict[Hashable, MetaAnalysisResult],
) -> tuple[float, int, float, float, tuple[str, ...]]:
    standard_errors = np.asarray(
        [group.standard_error for group in groups.values()], dtype=np.float64
    )
    if np.any(~np.isfinite(standard_errors)) or np.any(standard_errors <= 0.0):
        warning = (
            "The subgroup-differences test is unavailable because at least one "
            "subgroup has a non-finite or non-positive pooled standard error."
        )
        return float("nan"), len(groups) - 1, float("nan"), float("nan"), (warning,)

    estimates = np.asarray(
        [group.estimate for group in groups.values()], dtype=np.float64
    )
    weights = 1.0 / standard_errors**2
    pooled = float(np.dot(weights, estimates) / np.sum(weights))
    q_between = float(np.dot(weights, (estimates - pooled) ** 2))
    df = len(groups) - 1
    pvalue = float(chi2.sf(q_between, df))
    i2 = 0.0 if q_between <= 0.0 else max(0.0, (q_between - df) / q_between)
    return q_between, df, pvalue, i2, ()


def fit_subgroup_analysis(
    *,
    data: pd.DataFrame | None,
    subgroup: ColumnOrArray,
    overall: MetaAnalysisResult,
    fit_group: GroupFitter,
) -> SubgroupMetaAnalysisResult:
    """Fit subgroups and compare their pooled effects using a Wald Q test.

    The comparison follows the RevMan formulation: subgroup summary effects
    are weighted by the inverse square of their pooled standard errors and
    synthesized under a fixed-effect model. Random-effects tau-squared values
    are estimated independently within each subgroup.
    """

    overall_studies = overall.study_results
    labels = _subgroup_labels(subgroup, data=data, length=len(overall_studies))
    try:
        codes, unique_labels = pd.factorize(labels, sort=False)
    except TypeError as error:
        raise InvalidStudyDataError(
            "subgroup labels must be hashable scalars."
        ) from error

    if len(unique_labels) < 2:
        raise InsufficientStudiesError(
            "Subgroup analysis requires at least two distinct subgroup labels."
        )

    included = overall_studies["included"].to_numpy(dtype=np.bool_, copy=True)
    groups: dict[Hashable, MetaAnalysisResult] = {}
    for code, raw_label in enumerate(unique_labels):
        positions = np.flatnonzero(codes == code).astype(np.int64, copy=False)
        label = cast(Hashable, raw_label)
        if not np.any(included[positions]):
            raise InsufficientStudiesError(
                f"Subgroup {label!r} has no included studies after exclusions."
            )
        try:
            group_result = fit_group(positions)
        except InsufficientStudiesError as error:
            raise InsufficientStudiesError(f"Subgroup {label!r}: {error}") from error
        groups[label] = _restore_global_rows(
            group_result,
            positions=positions,
            label=label,
            source_data=data,
        )

    q_between, df, pvalue, i2, test_warnings = _between_group_test(groups)
    combined_studies = overall_studies.copy(deep=True)
    combined_studies.insert(2, "subgroup", labels)
    method = SubgroupMethodConfig(
        model=overall.model,
        tau2_strategy="independent" if overall.model == "random" else "not_applicable",
        test_method="fixed_effect_on_subgroup_estimates",
        subgroup_missing="raise",
    )
    return SubgroupMetaAnalysisResult(
        groups=groups,
        overall=overall,
        q_between=q_between,
        q_between_df=df,
        q_between_pvalue=pvalue,
        i2_between=i2,
        method=method,
        warnings=test_warnings,
        _study_results=combined_studies,
    )
