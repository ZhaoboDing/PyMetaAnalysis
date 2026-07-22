"""Deterministic moderator normalization and design-matrix construction."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data import (
    ColumnOrArray,
    MissingPolicy,
    _resolve_vector,
    normalize_studies,
)
from .exceptions import InsufficientStudiesError, InvalidStudyDataError

ModeratorInput: TypeAlias = Sequence[str] | Mapping[str, ColumnOrArray]
CategoricalInput: TypeAlias = Mapping[str, Sequence[Hashable]]

_RESERVED_TERM_NAMES = frozenset({"intercept"})


@dataclass(frozen=True, slots=True)
class ModeratorSpec:
    """One moderator and its resolved numerical encoding."""

    name: str
    kind: Literal["numeric", "categorical"]
    term_names: tuple[str, ...]
    levels: tuple[Hashable, ...] = ()
    reference: Hashable | None = None


@dataclass(frozen=True, slots=True)
class DesignInfo:
    """Replayable description of a fitted meta-regression design matrix."""

    intercept: bool
    moderators: tuple[ModeratorSpec, ...]
    term_names: tuple[str, ...]

    @property
    def moderator_names(self) -> tuple[str, ...]:
        """Return original moderator names in their declared order."""

        return tuple(spec.name for spec in self.moderators)

    def terms_for(self, moderator: str) -> tuple[str, ...]:
        """Return encoded terms belonging to one original moderator."""

        for spec in self.moderators:
            if spec.name == moderator:
                return spec.term_names
        raise InvalidStudyDataError(f"Unknown moderator {moderator!r}.")


@dataclass(frozen=True, slots=True)
class NormalizedMetaRegressionData:
    """Validated study rows, moderators, and their encoded design matrix."""

    row_id: NDArray[np.int64]
    study: NDArray[np.object_]
    effect: NDArray[np.float64]
    variance: NDArray[np.float64]
    included: NDArray[np.bool_]
    exclusion_reason: NDArray[np.object_]
    moderator_values: tuple[tuple[str, NDArray[Any]], ...]
    design_info: DesignInfo
    design_matrix: NDArray[np.float64]
    condition_number: float

    @property
    def included_effect(self) -> NDArray[np.float64]:
        return self.effect[self.included]

    @property
    def included_variance(self) -> NDArray[np.float64]:
        return self.variance[self.included]

    @property
    def included_design_matrix(self) -> NDArray[np.float64]:
        return self.design_matrix[self.included]


@dataclass(frozen=True, slots=True)
class _RawModerator:
    name: str
    values: NDArray[Any]
    missing: NDArray[np.bool_]
    levels: tuple[Hashable, ...] | None


def _moderator_mapping(
    moderators: ModeratorInput,
) -> tuple[tuple[str, ColumnOrArray], ...]:
    if isinstance(moderators, Mapping):
        items = tuple(moderators.items())
    else:
        if isinstance(moderators, str | bytes):
            raise InvalidStudyDataError(
                "moderators must be a sequence of column names or a "
                "name-to-input mapping."
            )
        items = tuple((name, name) for name in moderators)

    if not items:
        raise InvalidStudyDataError("At least one moderator must be provided.")

    names: list[str] = []
    for name, _ in items:
        if not isinstance(name, str) or not name:
            raise InvalidStudyDataError("Moderator names must be non-empty strings.")
        if name in _RESERVED_TERM_NAMES:
            raise InvalidStudyDataError(
                f"Moderator name {name!r} is reserved for the model intercept."
            )
        if name in names:
            raise InvalidStudyDataError(f"Duplicate moderator name {name!r}.")
        names.append(name)
    return items


def _is_missing(values: NDArray[Any]) -> NDArray[np.bool_]:
    missing = np.asarray(pd.isna(values), dtype=np.bool_)
    if missing.ndim != 1:
        raise InvalidStudyDataError("Moderator values must be scalar values.")
    return missing


def _scalar_is_missing(value: Any) -> bool:
    missing = pd.isna(value)
    if np.ndim(missing) != 0:
        raise ValueError("categorical levels and values must be scalar")
    return bool(missing)


def _validate_levels(name: str, levels: Sequence[Hashable]) -> tuple[Hashable, ...]:
    if isinstance(levels, str | bytes):
        raise InvalidStudyDataError(
            f"categorical[{name!r}] must be an ordered sequence of levels."
        )
    resolved = tuple(levels)
    if len(resolved) < 2:
        raise InvalidStudyDataError(
            f"Categorical moderator {name!r} requires at least two levels."
        )

    seen: set[Hashable] = set()
    for level in resolved:
        try:
            if _scalar_is_missing(level):
                raise InvalidStudyDataError(
                    f"Categorical moderator {name!r} has a missing level."
                )
            hash(level)
        except (TypeError, ValueError):
            raise InvalidStudyDataError(
                f"Levels for categorical moderator {name!r} must be non-missing "
                "hashable scalar values."
            ) from None
        if level in seen:
            raise InvalidStudyDataError(
                f"Categorical moderator {name!r} has duplicate level {level!r}."
            )
        seen.add(level)
    return resolved


def _equal(value: Any, level: Hashable) -> bool:
    try:
        comparison = value == level
        return bool(comparison) if np.ndim(comparison) == 0 else False
    except (TypeError, ValueError):
        return False


def _category_codes(
    values: NDArray[Any], levels: tuple[Hashable, ...]
) -> NDArray[np.int64]:
    codes = np.full(len(values), -1, dtype=np.int64)
    for position, value in enumerate(values):
        if _scalar_is_missing(value):
            continue
        for code, level in enumerate(levels):
            if _equal(value, level):
                codes[position] = code
                break
    return codes


def _numeric_values(name: str, values: NDArray[Any]) -> NDArray[np.float64]:
    nonmissing = values[~_is_missing(values)]
    if any(isinstance(value, str | bytes) for value in nonmissing):
        raise InvalidStudyDataError(
            f"Moderator {name!r} contains string values; declare it in categorical."
        )
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidStudyDataError(
            f"Numeric moderator {name!r} must contain numeric values."
        ) from error
    missing = np.isnan(numeric)
    invalid = (~missing) & (~np.isfinite(numeric))
    if np.any(invalid):
        rows = np.flatnonzero(invalid).tolist()
        raise InvalidStudyDataError(
            f"Numeric moderator {name!r} must be finite; invalid rows: {rows}."
        )
    return numeric


def _term_name(name: str, level: Hashable) -> str:
    return f"{name}[{level}]"


def normalize_meta_regression_data(
    *,
    data: pd.DataFrame | None,
    effect: ColumnOrArray,
    variance: ColumnOrArray | None,
    standard_error: ColumnOrArray | None,
    moderators: ModeratorInput,
    categorical: CategoricalInput | None,
    study: ColumnOrArray | None,
    missing: MissingPolicy,
    intercept: bool,
) -> NormalizedMetaRegressionData:
    """Resolve study inputs and build a full-rank moderator design matrix."""

    if not isinstance(intercept, bool):
        raise InvalidStudyDataError("intercept must be a boolean.")
    if missing not in {"raise", "drop"}:
        raise InvalidStudyDataError("missing must be either 'raise' or 'drop'.")

    moderator_items = _moderator_mapping(moderators)
    categorical = {} if categorical is None else categorical
    if not isinstance(categorical, Mapping):
        raise InvalidStudyDataError("categorical must be a mapping or None.")
    unknown_categorical = set(categorical) - {name for name, _ in moderator_items}
    if unknown_categorical:
        unknown_names = sorted(str(name) for name in unknown_categorical)
        raise InvalidStudyDataError(
            f"categorical contains unknown moderator(s): {unknown_names}."
        )
    if not intercept and categorical:
        raise InvalidStudyDataError(
            "intercept=False is only supported with numeric moderators in version 0.3."
        )

    # Use the existing validation path while postponing missing-value rejection
    # until moderator and study fields can be reported together.
    studies = normalize_studies(
        data=data,
        effect=effect,
        variance=variance,
        standard_error=standard_error,
        study=study,
        missing="drop",
    )
    row_count = len(studies.row_id)

    raw_moderators: list[_RawModerator] = []
    for name, value in moderator_items:
        raw = _resolve_vector(value, data=data, name=f"moderator {name!r}")
        if len(raw) != row_count:
            raise InvalidStudyDataError(
                f"Moderator {name!r} has length {len(raw)}, expected {row_count}."
            )
        if data is not None and len(raw) != len(data):
            raise InvalidStudyDataError(
                f"Moderator {name!r} must have one value per DataFrame row."
            )
        missing_values = _is_missing(raw)
        levels = (
            _validate_levels(name, categorical[name]) if name in categorical else None
        )
        if levels is None:
            raw = _numeric_values(name, raw)
        else:
            codes = _category_codes(raw, levels)
            unknown_levels = (~missing_values) & (codes < 0)
            if np.any(unknown_levels):
                rows = np.flatnonzero(unknown_levels).tolist()
                raise InvalidStudyDataError(
                    f"Categorical moderator {name!r} contains undeclared levels "
                    f"at rows {rows}."
                )
            raw = np.asarray(raw, dtype=object)
        raw_moderators.append(_RawModerator(name, raw, missing_values, levels))

    study_missing = _is_missing(studies.study)
    effect_missing = np.isnan(studies.effect)
    uncertainty_missing = np.isnan(studies.variance)
    any_missing = effect_missing | uncertainty_missing | study_missing
    for moderator in raw_moderators:
        any_missing |= moderator.missing

    uncertainty_label = "standard error" if standard_error is not None else "variance"
    reasons = np.full(row_count, None, dtype=object)
    for row in np.flatnonzero(any_missing):
        fields: list[str] = []
        if effect_missing[row]:
            fields.append("effect")
        if uncertainty_missing[row]:
            fields.append(uncertainty_label)
        if study_missing[row]:
            fields.append("study")
        fields.extend(
            f"moderator {moderator.name!r}"
            for moderator in raw_moderators
            if moderator.missing[row]
        )
        reasons[row] = "missing " + " and ".join(fields)

    if np.any(any_missing) and missing == "raise":
        details = {int(row): str(reasons[row]) for row in np.flatnonzero(any_missing)}
        raise InvalidStudyDataError(
            f"Missing meta-regression inputs by row: {details}; use "
            "missing='drop' to exclude them explicitly."
        )
    included = ~any_missing
    if not np.any(included):
        raise InvalidStudyDataError(
            "No studies remain after applying the missing-value policy."
        )

    specs: list[ModeratorSpec] = []
    columns: list[NDArray[np.float64]] = []
    term_names: list[str] = []
    if intercept:
        term_names.append("intercept")
        columns.append(np.ones(int(np.count_nonzero(included)), dtype=np.float64))

    for moderator in raw_moderators:
        if moderator.levels is None:
            terms: tuple[str, ...] = (moderator.name,)
            columns.append(np.asarray(moderator.values[included], dtype=np.float64))
            term_names.extend(terms)
            specs.append(ModeratorSpec(moderator.name, "numeric", terms))
            continue

        codes = _category_codes(moderator.values, moderator.levels)
        included_codes = codes[included]
        absent = [
            level
            for code, level in enumerate(moderator.levels)
            if not np.any(included_codes == code)
        ]
        if absent:
            raise InvalidStudyDataError(
                f"Categorical moderator {moderator.name!r} has declared levels "
                f"absent after exclusions: {absent!r}."
            )
        terms = tuple(
            _term_name(moderator.name, level) for level in moderator.levels[1:]
        )
        term_names.extend(terms)
        columns.extend(
            np.asarray(included_codes == code, dtype=np.float64)
            for code in range(1, len(moderator.levels))
        )
        specs.append(
            ModeratorSpec(
                moderator.name,
                "categorical",
                terms,
                levels=moderator.levels,
                reference=moderator.levels[0],
            )
        )

    if len(set(term_names)) != len(term_names):
        raise InvalidStudyDataError(
            "Moderator encoding produced duplicate term names; rename moderators "
            "or categorical levels."
        )

    design = np.column_stack(columns).astype(np.float64, copy=False)
    k, p = design.shape
    if k <= p:
        raise InsufficientStudiesError(
            f"Meta-regression requires k > p; got {k} included studies and {p} "
            "model coefficients."
        )
    rank = int(np.linalg.matrix_rank(design))
    if rank != p:
        raise InvalidStudyDataError(
            f"Meta-regression design matrix is rank deficient (rank {rank}, p={p})."
        )
    condition_number = float(np.linalg.cond(design))

    full_design = np.full((row_count, p), np.nan, dtype=np.float64)
    full_design[included] = design
    return NormalizedMetaRegressionData(
        row_id=studies.row_id,
        study=studies.study,
        effect=studies.effect,
        variance=studies.variance,
        included=included,
        exclusion_reason=reasons,
        moderator_values=tuple(
            (moderator.name, moderator.values.copy()) for moderator in raw_moderators
        ),
        design_info=DesignInfo(intercept, tuple(specs), tuple(term_names)),
        design_matrix=full_design,
        condition_number=condition_number,
    )


def build_prediction_design_matrix(
    data: pd.DataFrame,
    design_info: DesignInfo,
) -> NDArray[np.float64]:
    """Encode new moderator rows using a fitted design specification."""

    if not isinstance(data, pd.DataFrame):
        raise InvalidStudyDataError("new_data must be a pandas DataFrame.")
    if len(data) == 0:
        raise InvalidStudyDataError("new_data must contain at least one row.")

    columns: list[NDArray[np.float64]] = []
    if design_info.intercept:
        columns.append(np.ones(len(data), dtype=np.float64))

    for spec in design_info.moderators:
        if spec.name not in data.columns:
            raise InvalidStudyDataError(
                f"new_data is missing moderator column {spec.name!r}."
            )
        raw = data[spec.name].to_numpy(copy=True)
        missing = _is_missing(raw)
        if np.any(missing):
            rows = np.flatnonzero(missing).tolist()
            raise InvalidStudyDataError(
                f"new_data moderator {spec.name!r} is missing at rows {rows}."
            )
        if spec.kind == "numeric":
            columns.append(_numeric_values(spec.name, raw))
            continue

        codes = _category_codes(raw, spec.levels)
        unknown = codes < 0
        if np.any(unknown):
            rows = np.flatnonzero(unknown).tolist()
            raise InvalidStudyDataError(
                f"new_data moderator {spec.name!r} has unknown levels at rows {rows}."
            )
        columns.extend(
            np.asarray(codes == code, dtype=np.float64)
            for code in range(1, len(spec.levels))
        )

    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    if matrix.shape[1] != len(design_info.term_names):  # pragma: no cover
        raise RuntimeError("Prediction design does not match fitted terms.")
    return matrix
