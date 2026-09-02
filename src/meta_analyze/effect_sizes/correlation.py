"""Fisher's r-to-z effect sizes for independent correlations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..data import (
    ColumnOrArray,
    MissingPolicy,
    _resolve_vector,
    _study_labels,
    _validate_finite_precision_variance,
)
from ..exceptions import InvalidStudyDataError, UnsupportedMethodError


@dataclass(frozen=True, slots=True)
class CorrelationStudies:
    """Validated correlation coefficients and sample sizes."""

    row_id: NDArray[np.int64]
    study: NDArray[np.object_]
    correlation: NDArray[np.float64]
    n: NDArray[np.float64]
    included: NDArray[np.bool_]
    exclusion_reason: NDArray[np.object_]


@dataclass(frozen=True, slots=True)
class CorrelationEffectData:
    """Fisher's z effects and sampling variances."""

    studies: CorrelationStudies
    effect: NDArray[np.float64]
    variance: NDArray[np.float64]
    measure: str = "ZCOR"
    effect_scale: str = "fisher_z"
    display_scale: str = "tanh"

    @property
    def included_effect(self) -> NDArray[np.float64]:
        return self.effect[self.studies.included]

    @property
    def included_variance(self) -> NDArray[np.float64]:
        return self.variance[self.studies.included]


def _missing_reason(names: list[str]) -> str:
    return "missing " + ", ".join(names)


def normalize_correlation_studies(
    *,
    data: pd.DataFrame | None,
    correlation: ColumnOrArray,
    n: ColumnOrArray,
    study: ColumnOrArray | None,
    missing: MissingPolicy,
) -> CorrelationStudies:
    """Resolve and validate correlations and their study sample sizes."""

    if data is not None and not isinstance(data, pd.DataFrame):
        raise InvalidStudyDataError("data must be a pandas DataFrame or None.")
    if missing not in {"raise", "drop"}:
        raise InvalidStudyDataError("missing must be either 'raise' or 'drop'.")

    raw_correlation = _resolve_vector(
        correlation,
        data=data,
        name="correlation",
    )
    raw_n = _resolve_vector(n, data=data, name="n")
    if len(raw_correlation) != len(raw_n):
        raise InvalidStudyDataError(
            "correlation and n must have the same length; "
            f"got {len(raw_correlation)} and {len(raw_n)}."
        )
    length = len(raw_correlation)
    if data is not None and len(data) != length:
        raise InvalidStudyDataError(
            "Array-like correlation inputs used with data must have exactly one "
            "value per DataFrame row."
        )
    if length == 0:
        raise InvalidStudyDataError("At least one study row is required.")
    labels = _study_labels(
        study,
        data=data,
        length=length,
        uncertainty_label="sample size",
    )

    try:
        correlation_values = np.asarray(raw_correlation, dtype=np.float64)
        n_values = np.asarray(raw_n, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidStudyDataError(
            "correlation and n must contain numeric values."
        ) from error

    correlation_missing = pd.isna(correlation_values)
    n_missing = pd.isna(n_values)
    any_missing = correlation_missing | n_missing
    if np.any(any_missing) and missing == "raise":
        rows = np.flatnonzero(any_missing).tolist()
        raise InvalidStudyDataError(
            f"Missing correlation inputs at row positions {rows}; use "
            "missing='drop' to exclude them explicitly."
        )

    for name, values, missing_values in (
        ("correlation", correlation_values, correlation_missing),
        ("n", n_values, n_missing),
    ):
        invalid = ~missing_values & ~np.isfinite(values)
        if np.any(invalid):
            rows = np.flatnonzero(invalid).tolist()
            raise InvalidStudyDataError(f"{name} must be finite; invalid rows: {rows}.")

    included = ~any_missing
    invalid_correlation = included & (
        (correlation_values <= -1.0) | (correlation_values >= 1.0)
    )
    if np.any(invalid_correlation):
        rows = np.flatnonzero(invalid_correlation).tolist()
        raise InvalidStudyDataError(
            "correlation must be strictly between -1 and 1 for Fisher's z "
            f"transformation; invalid rows: {rows}."
        )

    noninteger_n = included & (n_values != np.floor(n_values))
    if np.any(noninteger_n):
        rows = np.flatnonzero(noninteger_n).tolist()
        raise InvalidStudyDataError(
            f"n must contain whole-number sample sizes; invalid rows: {rows}."
        )
    too_small_n = included & (n_values < 4.0)
    if np.any(too_small_n):
        rows = np.flatnonzero(too_small_n).tolist()
        raise InvalidStudyDataError(
            "n must be at least 4 so the Fisher's z sampling variance "
            f"1 / (n - 3) is finite and positive; invalid rows: {rows}."
        )

    reasons = np.full(length, None, dtype=object)
    for index in np.flatnonzero(any_missing):
        missing_names: list[str] = []
        if correlation_missing[index]:
            missing_names.append("correlation")
        if n_missing[index]:
            missing_names.append("n")
        reasons[index] = _missing_reason(missing_names)
    if not np.any(included):
        raise InvalidStudyDataError(
            "No studies remain after applying the missing-value policy."
        )

    return CorrelationStudies(
        row_id=np.arange(length, dtype=np.int64),
        study=labels,
        correlation=correlation_values,
        n=n_values,
        included=included,
        exclusion_reason=reasons,
    )


def calculate_correlation_effects(
    studies: CorrelationStudies,
    *,
    measure: str,
) -> CorrelationEffectData:
    """Calculate Fisher's z correlations and ``1 / (n - 3)`` variances."""

    if not isinstance(measure, str) or measure.upper() != "ZCOR":
        raise UnsupportedMethodError(
            "measure currently supports only 'ZCOR' (Fisher's r-to-z transformation)."
        )

    included = studies.included
    effect = np.full(len(included), np.nan, dtype=np.float64)
    variance = np.full(len(included), np.nan, dtype=np.float64)
    effect[included] = np.arctanh(studies.correlation[included])
    variance[included] = 1.0 / (studies.n[included] - 3.0)

    invalid_effect = included & ~np.isfinite(effect)
    if np.any(invalid_effect):  # pragma: no cover - guarded by input bounds
        rows = np.flatnonzero(invalid_effect).tolist()
        raise InvalidStudyDataError(
            "Fisher's z effect sizes must be finite after transformation; "
            f"invalid rows: {rows}."
        )
    invalid_variance = included & (~np.isfinite(variance) | (variance <= 0.0))
    if np.any(invalid_variance):  # pragma: no cover - guarded by sample-size checks
        rows = np.flatnonzero(invalid_variance).tolist()
        raise InvalidStudyDataError(
            "Fisher's z sampling variances must be finite and strictly positive; "
            f"invalid rows: {rows}."
        )
    _validate_finite_precision_variance(
        variance,
        included=included,
        label="Fisher's z sampling variances",
    )

    return CorrelationEffectData(
        studies=studies,
        effect=effect,
        variance=variance,
    )
