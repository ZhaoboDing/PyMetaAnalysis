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
    uncertainty_label: str = "variance",
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
            f"study has length {len(labels)}, but effect and {uncertainty_label} "
            f"have length {length}."
        )
    return np.asarray(labels, dtype=object)


def _select_uncertainty_input(
    variance: ColumnOrArray | None,
    standard_error: ColumnOrArray | None,
) -> tuple[ColumnOrArray, str, str]:
    if (variance is None) == (standard_error is None):
        raise InvalidStudyDataError(
            "Exactly one of variance or standard_error must be provided."
        )
    if standard_error is not None:
        return standard_error, "standard_error", "standard error"
    if variance is None:  # pragma: no cover - guarded by the exclusive check
        raise RuntimeError("variance input unexpectedly missing")
    return variance, "variance", "variance"


def normalize_studies(
    *,
    data: pd.DataFrame | None,
    effect: ColumnOrArray,
    variance: ColumnOrArray | None,
    standard_error: ColumnOrArray | None = None,
    study: ColumnOrArray | None,
    missing: MissingPolicy,
) -> NormalizedStudies:
    """Resolve column/array arguments and validate generic effect data."""

    if data is not None and not isinstance(data, pd.DataFrame):
        raise InvalidStudyDataError("data must be a pandas DataFrame or None.")
    if missing not in {"raise", "drop"}:
        raise InvalidStudyDataError("missing must be either 'raise' or 'drop'.")

    uncertainty, uncertainty_name, uncertainty_label = _select_uncertainty_input(
        variance, standard_error
    )
    raw_effect = _resolve_vector(effect, data=data, name="effect")
    raw_uncertainty = _resolve_vector(uncertainty, data=data, name=uncertainty_name)
    if len(raw_effect) != len(raw_uncertainty):
        raise InvalidStudyDataError(
            f"effect and {uncertainty_label} must have the same length; "
            f"got {len(raw_effect)} and {len(raw_uncertainty)}."
        )
    if data is not None and len(data) != len(raw_effect):
        raise InvalidStudyDataError(
            "Array-like inputs used with data must have exactly one value per "
            "DataFrame row."
        )

    labels = _study_labels(
        study,
        data=data,
        length=len(raw_effect),
        uncertainty_label=uncertainty_label,
    )

    try:
        effect_values = np.asarray(raw_effect, dtype=np.float64)
        uncertainty_values = np.asarray(raw_uncertainty, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidStudyDataError(
            f"effect and {uncertainty_label} must contain numeric values."
        ) from error

    effect_missing = pd.isna(effect_values)
    uncertainty_missing = pd.isna(uncertainty_values)
    any_missing = effect_missing | uncertainty_missing
    if np.any(any_missing) and missing == "raise":
        rows = np.flatnonzero(any_missing).tolist()
        raise InvalidStudyDataError(
            f"Missing effect or {uncertainty_label} values at row positions {rows}; "
            "use missing='drop' to exclude them explicitly."
        )

    finite_effect = np.isfinite(effect_values) | effect_missing
    finite_uncertainty = np.isfinite(uncertainty_values) | uncertainty_missing
    if not np.all(finite_effect):
        rows = np.flatnonzero(~finite_effect).tolist()
        raise InvalidStudyDataError(
            f"Effect values must be finite; invalid rows: {rows}."
        )
    if not np.all(finite_uncertainty):
        rows = np.flatnonzero(~finite_uncertainty).tolist()
        raise InvalidStudyDataError(
            f"{uncertainty_label.capitalize()} values must be finite; "
            f"invalid rows: {rows}."
        )

    nonpositive_uncertainty = (~uncertainty_missing) & (uncertainty_values <= 0.0)
    if np.any(nonpositive_uncertainty):
        rows = np.flatnonzero(nonpositive_uncertainty).tolist()
        raise InvalidStudyDataError(
            f"Sampling {uncertainty_label}s must be strictly positive; "
            f"invalid rows: {rows}."
        )

    if uncertainty_name == "standard_error":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            variance_values = np.square(uncertainty_values)
        invalid_variance = (~uncertainty_missing) & (
            (~np.isfinite(variance_values)) | (variance_values <= 0.0)
        )
        if np.any(invalid_variance):
            rows = np.flatnonzero(invalid_variance).tolist()
            raise InvalidStudyDataError(
                "Standard errors must produce finite, strictly positive sampling "
                f"variances after squaring; invalid rows: {rows}."
            )
    else:
        variance_values = uncertainty_values

    included = ~any_missing
    reasons = np.full(len(effect_values), None, dtype=object)
    for index in np.flatnonzero(any_missing):
        if effect_missing[index] and uncertainty_missing[index]:
            reasons[index] = f"missing effect and {uncertainty_label}"
        elif effect_missing[index]:
            reasons[index] = "missing effect"
        else:
            reasons[index] = f"missing {uncertainty_label}"

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
