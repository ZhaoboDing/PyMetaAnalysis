"""Input normalization at the pandas/NumPy boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .exceptions import InvalidStudyDataError

ColumnOrArray: TypeAlias = str | ArrayLike
MissingPolicy: TypeAlias = Literal["raise", "drop"]


@dataclass(frozen=True, slots=True)
class NormalizedStudies:
    """Validated study vectors plus row-level inclusion metadata."""

    row_id: NDArray[np.int64]
    study: NDArray[np.object_]
    effect: NDArray[np.float64]
    variance: NDArray[np.float64]
    included: NDArray[np.bool_]
    exclusion_reason: NDArray[np.object_]

    @property
    def included_effect(self) -> NDArray[np.float64]:
        return self.effect[self.included]

    @property
    def included_variance(self) -> NDArray[np.float64]:
        return self.variance[self.included]


def _resolve_vector(
    value: ColumnOrArray,
    *,
    data: pd.DataFrame | None,
    name: str,
) -> NDArray[Any]:
    if isinstance(value, str):
        if data is None:
            raise InvalidStudyDataError(
                f"{name}={value!r} is a column name, but no DataFrame was provided."
            )
        if value not in data.columns:
            raise InvalidStudyDataError(
                f"Column {value!r}, specified by {name}, is not present in data."
            )
        array = data[value].to_numpy(copy=True)
    else:
        if isinstance(value, (str, bytes)):
            raise InvalidStudyDataError(
                f"{name} must be a column name or 1D array-like."
            )
        array = np.asarray(value)

    if array.ndim != 1:
        raise InvalidStudyDataError(
            f"{name} must be one-dimensional, got shape {array.shape}."
        )
    return array


def _study_labels(
    study: ColumnOrArray | None,
    *,
    data: pd.DataFrame | None,
    length: int,
) -> NDArray[np.object_]:
    if study is None:
        labels: NDArray[Any]
        if data is not None:
            labels = data.index.to_numpy(copy=True)
        else:
            labels = np.arange(length, dtype=np.int64)
    else:
        labels = _resolve_vector(study, data=data, name="study")

    if len(labels) != length:
        raise InvalidStudyDataError(
            f"study has length {len(labels)}, but effect and variance have "
            f"length {length}."
        )
    return np.asarray(labels, dtype=object)


def normalize_studies(
    *,
    data: pd.DataFrame | None,
    effect: ColumnOrArray,
    variance: ColumnOrArray,
    study: ColumnOrArray | None,
    missing: MissingPolicy,
) -> NormalizedStudies:
    """Resolve column/array arguments and validate generic effect data."""

    if data is not None and not isinstance(data, pd.DataFrame):
        raise InvalidStudyDataError("data must be a pandas DataFrame or None.")
    if missing not in {"raise", "drop"}:
        raise InvalidStudyDataError("missing must be either 'raise' or 'drop'.")

    raw_effect = _resolve_vector(effect, data=data, name="effect")
    raw_variance = _resolve_vector(variance, data=data, name="variance")
    if len(raw_effect) != len(raw_variance):
        raise InvalidStudyDataError(
            "effect and variance must have the same length; "
            f"got {len(raw_effect)} and {len(raw_variance)}."
        )
    if data is not None and len(data) != len(raw_effect):
        raise InvalidStudyDataError(
            "Array-like inputs used with data must have exactly one value per "
            "DataFrame row."
        )

    labels = _study_labels(study, data=data, length=len(raw_effect))

    try:
        effect_values = np.asarray(raw_effect, dtype=np.float64)
        variance_values = np.asarray(raw_variance, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidStudyDataError(
            "effect and variance must contain numeric values."
        ) from error

    effect_missing = pd.isna(effect_values)
    variance_missing = pd.isna(variance_values)
    any_missing = effect_missing | variance_missing
    if np.any(any_missing) and missing == "raise":
        rows = np.flatnonzero(any_missing).tolist()
        raise InvalidStudyDataError(
            f"Missing effect or variance values at row positions {rows}; "
            "use missing='drop' to exclude them explicitly."
        )

    finite_effect = np.isfinite(effect_values) | effect_missing
    finite_variance = np.isfinite(variance_values) | variance_missing
    if not np.all(finite_effect):
        rows = np.flatnonzero(~finite_effect).tolist()
        raise InvalidStudyDataError(
            f"Effect values must be finite; invalid rows: {rows}."
        )
    if not np.all(finite_variance):
        rows = np.flatnonzero(~finite_variance).tolist()
        raise InvalidStudyDataError(
            f"Variance values must be finite; invalid rows: {rows}."
        )

    nonpositive_variance = (~variance_missing) & (variance_values <= 0.0)
    if np.any(nonpositive_variance):
        rows = np.flatnonzero(nonpositive_variance).tolist()
        raise InvalidStudyDataError(
            f"Sampling variances must be strictly positive; invalid rows: {rows}."
        )

    included = ~any_missing
    reasons = np.full(len(effect_values), None, dtype=object)
    for index in np.flatnonzero(any_missing):
        if effect_missing[index] and variance_missing[index]:
            reasons[index] = "missing effect and variance"
        elif effect_missing[index]:
            reasons[index] = "missing effect"
        else:
            reasons[index] = "missing variance"

    if not np.any(included):
        raise InvalidStudyDataError(
            "No studies remain after applying the missing-value policy."
        )

    return NormalizedStudies(
        row_id=np.arange(len(effect_values), dtype=np.int64),
        study=labels,
        effect=effect_values,
        variance=variance_values,
        included=included,
        exclusion_reason=reasons,
    )
