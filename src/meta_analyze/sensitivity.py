"""Leave-one-out and cumulative meta-analysis workflows."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data import ColumnOrArray, MissingPolicy
from .exceptions import (
    InsufficientStudiesError,
    InvalidStudyDataError,
    UnsupportedMethodError,
)
from .provenance import remap_provenance_rows
from .results import MetaAnalysisResult, SubgroupMetaAnalysisResult


@dataclass(frozen=True, slots=True)
class LeaveOneOutResult:
    """Repeated fits obtained by omitting each included study in turn."""

    original: MetaAnalysisResult
    results: tuple[MetaAnalysisResult, ...]
    warnings: tuple[str, ...]
    _table: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_table", self._table.copy(deep=True))

    def __len__(self) -> int:
        return len(self.results)

    @property
    def table(self) -> pd.DataFrame:
        """Return one row per omitted study and refitted model."""

        return self._table.copy(deep=True)

    def summary(self) -> pd.DataFrame:
        """Return the tabular leave-one-out summary."""

        return self.table

    def to_dataframe(self) -> pd.DataFrame:
        """Return the tabular leave-one-out summary."""

        return self.table


@dataclass(frozen=True, slots=True)
class CumulativeMetaAnalysisResult:
    """Repeated fits obtained while accumulating studies in a stable order."""

    original: MetaAnalysisResult
    results: tuple[MetaAnalysisResult, ...]
    warnings: tuple[str, ...]
    _table: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_table", self._table.copy(deep=True))

    def __len__(self) -> int:
        return len(self.results)

    @property
    def final(self) -> MetaAnalysisResult:
        """Return the final fit containing every originally included study."""

        return self.results[-1]

    @property
    def table(self) -> pd.DataFrame:
        """Return one row per estimable cumulative fit."""

        return self._table.copy(deep=True)

    def summary(self) -> pd.DataFrame:
        """Return the tabular cumulative summary."""

        return self.table

    def to_dataframe(self) -> pd.DataFrame:
        """Return the tabular cumulative summary."""

        return self.table


@dataclass(frozen=True, slots=True)
class SubgroupLeaveOneOutResult:
    """Leave-one-out results for each subgroup and the overall analysis."""

    groups: Mapping[Hashable, LeaveOneOutResult]
    overall: LeaveOneOutResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))

    def to_dataframe(self) -> pd.DataFrame:
        """Return subgroup and overall leave-one-out rows in one table."""

        return _combine_subgroup_tables(self.groups, self.overall)

    def summary(self) -> pd.DataFrame:
        """Return subgroup and overall leave-one-out rows in one table."""

        return self.to_dataframe()


@dataclass(frozen=True, slots=True)
class SubgroupCumulativeMetaAnalysisResult:
    """Cumulative results for each subgroup and the overall analysis."""

    groups: Mapping[Hashable, CumulativeMetaAnalysisResult]
    overall: CumulativeMetaAnalysisResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))

    def to_dataframe(self) -> pd.DataFrame:
        """Return subgroup and overall cumulative rows in one table."""

        return _combine_subgroup_tables(self.groups, self.overall)

    def summary(self) -> pd.DataFrame:
        """Return subgroup and overall cumulative rows in one table."""

        return self.to_dataframe()


def _combine_subgroup_tables(
    groups: Mapping[Hashable, LeaveOneOutResult | CumulativeMetaAnalysisResult],
    overall: LeaveOneOutResult | CumulativeMetaAnalysisResult,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, analysis in groups.items():
        table = analysis.to_dataframe()
        table.insert(0, "subgroup", [label for _ in range(len(table))])
        table.insert(0, "scope", "subgroup")
        frames.append(table)
    overall_table = overall.to_dataframe()
    overall_table.insert(0, "subgroup", None)
    overall_table.insert(0, "scope", "overall")
    frames.append(overall_table)
    return pd.concat(frames, ignore_index=True)


def _numeric_controls(result: MetaAnalysisResult) -> tuple[float, int]:
    return result.method.atol, result.method.max_iter


def _string_option(
    options: dict[str, str | float | int | bool | None],
    name: str,
    default: str,
) -> str:
    value = options.get(name, default)
    if not isinstance(value, str):
        raise RuntimeError(f"Stored {name} is not a string.")
    return value


def _float_option(
    options: dict[str, str | float | int | bool | None],
    name: str,
    default: float,
) -> float:
    value = options.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"Stored {name} is not numeric.")
    return float(value)


def _refit(
    result: MetaAnalysisResult, positions: NDArray[np.int64]
) -> MetaAnalysisResult:
    """Refit a stored model on selected local row positions."""

    from .api import meta_analysis
    from .binary_api import meta_binary
    from .continuous_api import meta_continuous

    studies = result.study_results
    selected = studies.iloc[positions]
    options = dict(result.method.options)
    atol, max_iter = _numeric_controls(result)
    tau2_method = result.method.tau2_method or "REML"
    missing = cast(MissingPolicy, result.method.missing)
    study = selected["study"].to_numpy(dtype=object, copy=True)

    fitted: MetaAnalysisResult
    if result.measure == "GENERIC":
        fitted = meta_analysis(
            effect=selected["effect"].to_numpy(dtype=np.float64, copy=True),
            variance=selected["variance"].to_numpy(dtype=np.float64, copy=True),
            study=study,
            model=result.model,
            tau2_method=tau2_method,
            ci_method=result.method.ci_method,
            confidence_level=result.method.confidence_level,
            missing=missing,
            atol=atol,
            max_iter=max_iter,
        )
    elif {"event_treat", "n_treat", "event_control", "n_control"}.issubset(
        selected.columns
    ):
        fitted = meta_binary(
            event_treat=selected["event_treat"].to_numpy(dtype=np.float64, copy=True),
            n_treat=selected["n_treat"].to_numpy(dtype=np.float64, copy=True),
            event_control=selected["event_control"].to_numpy(
                dtype=np.float64, copy=True
            ),
            n_control=selected["n_control"].to_numpy(dtype=np.float64, copy=True),
            measure=result.measure,
            method=result.method.pooling_method,
            continuity_correction=_float_option(options, "continuity_correction", 0.5),
            correction_scope=_string_option(
                options, "correction_scope", "only_zero_studies"
            ),
            mh_continuity_correction=_float_option(
                options, "mh_continuity_correction", 0.0
            ),
            mh_correction_scope=_string_option(
                options, "mh_correction_scope", "only_zero_studies"
            ),
            study=study,
            model=result.model,
            tau2_method=tau2_method,
            ci_method=result.method.ci_method,
            confidence_level=result.method.confidence_level,
            missing=missing,
            atol=atol,
            max_iter=max_iter,
        )
    elif {
        "mean_treat",
        "sd_treat",
        "n_treat",
        "mean_control",
        "sd_control",
        "n_control",
    }.issubset(selected.columns):
        fitted = meta_continuous(
            mean_treat=selected["mean_treat"].to_numpy(dtype=np.float64, copy=True),
            sd_treat=selected["sd_treat"].to_numpy(dtype=np.float64, copy=True),
            n_treat=selected["n_treat"].to_numpy(dtype=np.float64, copy=True),
            mean_control=selected["mean_control"].to_numpy(dtype=np.float64, copy=True),
            sd_control=selected["sd_control"].to_numpy(dtype=np.float64, copy=True),
            n_control=selected["n_control"].to_numpy(dtype=np.float64, copy=True),
            measure=result.measure,
            smd_variance=_string_option(options, "smd_variance", "LS"),
            study=study,
            model=result.model,
            tau2_method=tau2_method,
            ci_method=result.method.ci_method,
            confidence_level=result.method.confidence_level,
            missing=missing,
            atol=atol,
            max_iter=max_iter,
        )
    else:  # pragma: no cover - guarded by constructors in this package
        raise UnsupportedMethodError(
            f"Cannot reconstruct inputs for measure={result.measure!r}."
        )

    refitted_studies = fitted.study_results
    refitted_studies["row_id"] = selected["row_id"].to_numpy(copy=True)
    source = result.source_data
    selected_source = None if source is None else source.iloc[positions].copy(deep=True)
    return replace(
        fitted,
        provenance=remap_provenance_rows(
            fitted.provenance,
            selected["row_id"].to_numpy(dtype=np.int64, copy=True).tolist(),
        ),
        _study_results=refitted_studies,
        _source_data=selected_source,
    )


def _fit_summary(result: MetaAnalysisResult) -> dict[str, Any]:
    display_low, display_high = result.display_ci
    return {
        "k": result.k,
        "estimate": result.estimate,
        "standard_error": result.standard_error,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "display_estimate": result.display_estimate,
        "display_ci_low": display_low,
        "display_ci_high": display_high,
        "tau2": result.tau2,
        "q": result.q,
        "q_df": result.q_df,
        "q_pvalue": result.q_pvalue,
        "i2": result.i2,
        "h2": result.h2,
    }


def leave_one_out(result: MetaAnalysisResult) -> LeaveOneOutResult:
    """Refit the same model while omitting each included study once."""

    studies = result.study_results
    included = np.flatnonzero(
        studies["included"].to_numpy(dtype=np.bool_, copy=True)
    ).astype(np.int64, copy=False)
    minimum = 3 if result.model == "random" else 2
    if len(included) < minimum:
        requirement = (
            "three included studies for a random-effects model"
            if result.model == "random"
            else "two included studies"
        )
        raise InsufficientStudiesError(
            f"Leave-one-out analysis requires at least {requirement}."
        )

    fitted_results: list[MetaAnalysisResult] = []
    rows: list[dict[str, Any]] = []
    for omitted in included:
        retained = included[included != omitted]
        fitted = _refit(result, retained)
        fitted_results.append(fitted)
        source_row = studies.iloc[int(omitted)]
        rows.append(
            {
                "omitted_row_id": source_row["row_id"],
                "omitted_study": source_row["study"],
                **_fit_summary(fitted),
            }
        )

    return LeaveOneOutResult(
        original=result,
        results=tuple(fitted_results),
        warnings=(),
        _table=pd.DataFrame(rows),
    )


def _order_values(
    result: MetaAnalysisResult,
    order: ColumnOrArray | None,
) -> NDArray[np.object_]:
    studies = result.study_results
    row_count = len(studies)
    if order is None:
        return np.asarray(np.arange(row_count), dtype=object)
    if isinstance(order, str):
        source = result.source_data
        if source is not None and order in source.columns:
            values = source[order].to_numpy(copy=True)
        elif order in studies.columns:
            values = studies[order].to_numpy(copy=True)
        else:
            raise InvalidStudyDataError(
                f"Order column {order!r} is not present in the source data or "
                "study results."
            )
    else:
        values = np.asarray(order, dtype=object)
        if values.ndim != 1:
            raise InvalidStudyDataError(
                f"order must be one-dimensional, got shape {values.shape}."
            )
    if len(values) != row_count:
        raise InvalidStudyDataError(
            f"order has length {len(values)}, but the analysis has {row_count} rows."
        )
    return np.asarray(values, dtype=object)


def _ordered_batches(
    result: MetaAnalysisResult,
    *,
    order: ColumnOrArray | None,
    ascending: bool,
    collapse: bool,
) -> tuple[list[NDArray[np.int64]], NDArray[np.object_]]:
    if not isinstance(ascending, bool):
        raise InvalidStudyDataError("ascending must be a boolean.")
    if not isinstance(collapse, bool):
        raise InvalidStudyDataError("collapse must be a boolean.")
    if collapse and order is None:
        raise InvalidStudyDataError("collapse=True requires an explicit order.")

    studies = result.study_results
    included = np.flatnonzero(
        studies["included"].to_numpy(dtype=np.bool_, copy=True)
    ).astype(np.int64, copy=False)
    values = _order_values(result, order)
    included_values = values[included]
    missing = np.asarray(pd.isna(included_values), dtype=np.bool_)
    if np.any(missing):
        rows = included[np.flatnonzero(missing)].tolist()
        raise InvalidStudyDataError(
            f"Order values must not be missing for included studies; rows: {rows}."
        )

    sortable = pd.Series(included_values, index=included)
    try:
        sorted_series = sortable.sort_values(
            ascending=ascending,
            kind="mergesort",
        )
    except TypeError as error:
        raise InvalidStudyDataError(
            "Order values must be mutually sortable."
        ) from error
    sorted_positions = sorted_series.index.to_numpy(dtype=np.int64, copy=True)
    sorted_values = sorted_series.to_numpy(dtype=object, copy=True)
    if not collapse:
        return [
            np.asarray([position], dtype=np.int64) for position in sorted_positions
        ], values

    batches: list[NDArray[np.int64]] = []
    start = 0
    for index in range(1, len(sorted_positions) + 1):
        at_end = index == len(sorted_positions)
        changed = not at_end and sorted_values[index] != sorted_values[start]
        if at_end or changed:
            batches.append(sorted_positions[start:index])
            start = index
    return batches, values


def cumulative(
    result: MetaAnalysisResult,
    *,
    order: ColumnOrArray | None = None,
    ascending: bool = True,
    collapse: bool = False,
) -> CumulativeMetaAnalysisResult:
    """Refit the same model while adding studies in a stable order."""

    batches, order_values = _ordered_batches(
        result,
        order=order,
        ascending=ascending,
        collapse=collapse,
    )
    studies = result.study_results
    minimum = 2 if result.model == "random" else 1
    warnings: tuple[str, ...] = ()
    if minimum == 2:
        warnings = (
            "Random-effects cumulative analysis begins at k=2 because tau-squared "
            "is not estimable from a single study in this library.",
        )

    prefix: list[int] = []
    pending_added: list[int] = []
    fitted_results: list[MetaAnalysisResult] = []
    rows: list[dict[str, Any]] = []
    for batch in batches:
        batch_list = batch.tolist()
        prefix.extend(batch_list)
        pending_added.extend(batch_list)
        if len(prefix) < minimum:
            continue
        positions = np.asarray(prefix, dtype=np.int64)
        fitted = _refit(result, positions)
        fitted_results.append(fitted)
        added_rows = studies.iloc[pending_added]
        last_position = int(batch[-1])
        rows.append(
            {
                "step": len(fitted_results),
                "added_row_ids": tuple(added_rows["row_id"].tolist()),
                "added_studies": tuple(added_rows["study"].tolist()),
                "order_value": order_values[last_position],
                **_fit_summary(fitted),
            }
        )
        pending_added.clear()

    return CumulativeMetaAnalysisResult(
        original=result,
        results=tuple(fitted_results),
        warnings=warnings,
        _table=pd.DataFrame(rows),
    )


def subgroup_leave_one_out(
    result: SubgroupMetaAnalysisResult,
) -> SubgroupLeaveOneOutResult:
    """Run leave-one-out analyses within every subgroup and overall."""

    groups: dict[Hashable, LeaveOneOutResult] = {}
    for label, group in result.groups.items():
        try:
            groups[label] = leave_one_out(group)
        except InsufficientStudiesError as error:
            raise InsufficientStudiesError(f"Subgroup {label!r}: {error}") from error
    return SubgroupLeaveOneOutResult(
        groups=groups,
        overall=leave_one_out(result.overall),
    )


def subgroup_cumulative(
    result: SubgroupMetaAnalysisResult,
    *,
    order: ColumnOrArray | None = None,
    ascending: bool = True,
    collapse: bool = False,
) -> SubgroupCumulativeMetaAnalysisResult:
    """Run cumulative analyses within every subgroup and overall."""

    full_order = _order_values(result.overall, order)
    groups: dict[Hashable, CumulativeMetaAnalysisResult] = {}
    for label, group in result.groups.items():
        group_order: ColumnOrArray | None
        if order is None:
            group_order = None
        elif isinstance(order, str):
            group_order = order
        else:
            row_ids = group.study_results["row_id"].to_numpy(dtype=np.int64, copy=True)
            group_order = full_order[row_ids]
        groups[label] = cumulative(
            group,
            order=group_order,
            ascending=ascending,
            collapse=collapse,
        )
    return SubgroupCumulativeMetaAnalysisResult(
        groups=groups,
        overall=cumulative(
            result.overall,
            order=order,
            ascending=ascending,
            collapse=collapse,
        ),
    )
