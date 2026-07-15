"""Effect sizes and validation for two independent continuous groups.

The standardized mean difference follows ``metafor::escalc(measure="SMD")``:
the group mean difference is divided by the pooled standard deviation, then
multiplied by the exact Hedges correction. Its default ``LS`` sampling
variance is the large-sample approximation from Hedges (1982).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import gammaln

from ..data import ColumnOrArray, MissingPolicy, _resolve_vector, _study_labels
from ..exceptions import InvalidStudyDataError, UnsupportedMethodError


@dataclass(frozen=True, slots=True)
class ContinuousStudies:
    """Validated two-group summary statistics and inclusion metadata."""

    row_id: NDArray[np.int64]
    study: NDArray[np.object_]
    mean_treat: NDArray[np.float64]
    sd_treat: NDArray[np.float64]
    n_treat: NDArray[np.float64]
    mean_control: NDArray[np.float64]
    sd_control: NDArray[np.float64]
    n_control: NDArray[np.float64]
    included: NDArray[np.bool_]
    exclusion_reason: NDArray[np.object_]


@dataclass(frozen=True, slots=True)
class ContinuousEffectData:
    """Continuous study effects on their model scale."""

    studies: ContinuousStudies
    effect: NDArray[np.float64]
    variance: NDArray[np.float64]
    pooled_sd: NDArray[np.float64]
    cohen_d: NDArray[np.float64]
    correction_factor: NDArray[np.float64]
    measure: str
    effect_scale: str = "identity"
    display_scale: str = "identity"

    @property
    def included_effect(self) -> NDArray[np.float64]:
        return self.effect[self.studies.included]

    @property
    def included_variance(self) -> NDArray[np.float64]:
        return self.variance[self.studies.included]


def _missing_reason(names: list[str]) -> str:
    return "missing " + ", ".join(names)


def normalize_continuous_studies(
    *,
    data: pd.DataFrame | None,
    mean_treat: ColumnOrArray,
    sd_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    mean_control: ColumnOrArray,
    sd_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None,
    missing: MissingPolicy,
) -> ContinuousStudies:
    """Resolve and validate two-group continuous summary statistics."""

    if data is not None and not isinstance(data, pd.DataFrame):
        raise InvalidStudyDataError("data must be a pandas DataFrame or None.")
    if missing not in {"raise", "drop"}:
        raise InvalidStudyDataError("missing must be either 'raise' or 'drop'.")

    arguments = {
        "mean_treat": mean_treat,
        "sd_treat": sd_treat,
        "n_treat": n_treat,
        "mean_control": mean_control,
        "sd_control": sd_control,
        "n_control": n_control,
    }
    raw = {
        name: _resolve_vector(value, data=data, name=name)
        for name, value in arguments.items()
    }
    lengths = {len(values) for values in raw.values()}
    if len(lengths) != 1:
        detail = ", ".join(f"{name}={len(values)}" for name, values in raw.items())
        raise InvalidStudyDataError(
            f"Continuous inputs must have equal lengths; {detail}."
        )
    length = lengths.pop()
    if data is not None and len(data) != length:
        raise InvalidStudyDataError(
            "Array-like continuous inputs used with data must have exactly one "
            "value per DataFrame row."
        )
    labels = _study_labels(study, data=data, length=length)

    try:
        values = {
            name: np.asarray(vector, dtype=np.float64) for name, vector in raw.items()
        }
    except (TypeError, ValueError) as error:
        raise InvalidStudyDataError(
            "Continuous inputs must contain numeric values."
        ) from error

    missing_by_name = {name: pd.isna(vector) for name, vector in values.items()}
    any_missing = np.logical_or.reduce(tuple(missing_by_name.values()))
    if np.any(any_missing) and missing == "raise":
        rows = np.flatnonzero(any_missing).tolist()
        raise InvalidStudyDataError(
            f"Missing continuous inputs at row positions {rows}; use "
            "missing='drop' to exclude them explicitly."
        )

    for name, vector in values.items():
        present = ~missing_by_name[name]
        invalid = present & ~np.isfinite(vector)
        if np.any(invalid):
            rows = np.flatnonzero(invalid).tolist()
            raise InvalidStudyDataError(f"{name} must be finite; invalid rows: {rows}.")

    active = ~any_missing
    for name in ("n_treat", "n_control"):
        vector = values[name]
        noninteger = active & (vector != np.floor(vector))
        if np.any(noninteger):
            rows = np.flatnonzero(noninteger).tolist()
            raise InvalidStudyDataError(
                f"{name} must contain whole-number sample sizes; invalid rows: {rows}."
            )
        too_small = active & (vector < 2.0)
        if np.any(too_small):
            rows = np.flatnonzero(too_small).tolist()
            raise InvalidStudyDataError(
                f"{name} must be at least 2 when a sample SD is supplied; "
                f"invalid rows: {rows}."
            )

    for name in ("sd_treat", "sd_control"):
        invalid = active & (values[name] < 0.0)
        if np.any(invalid):
            rows = np.flatnonzero(invalid).tolist()
            raise InvalidStudyDataError(
                f"{name} must be non-negative; invalid rows: {rows}."
            )

    reasons = np.full(length, None, dtype=object)
    for index in np.flatnonzero(any_missing):
        names = [name for name, mask in missing_by_name.items() if mask[index]]
        reasons[index] = _missing_reason(names)
    if not np.any(active):
        raise InvalidStudyDataError(
            "No studies remain after applying the missing-value policy."
        )

    return ContinuousStudies(
        row_id=np.arange(length, dtype=np.int64),
        study=labels,
        mean_treat=values["mean_treat"],
        sd_treat=values["sd_treat"],
        n_treat=values["n_treat"],
        mean_control=values["mean_control"],
        sd_control=values["sd_control"],
        n_control=values["n_control"],
        included=active,
        exclusion_reason=reasons,
    )


def _exact_hedges_correction(df: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the exact gamma-function correction J(df)."""

    return np.exp(
        gammaln(df / 2.0) - 0.5 * np.log(df / 2.0) - gammaln((df - 1.0) / 2.0)
    )


def calculate_continuous_effects(
    studies: ContinuousStudies,
    *,
    measure: str,
    smd_variance: str = "LS",
) -> ContinuousEffectData:
    """Calculate mean differences or exact-corrected Hedges' g values."""

    normalized_measure = measure.upper()
    if normalized_measure not in {"MD", "SMD"}:
        raise UnsupportedMethodError("measure must be 'MD' or 'SMD'.")
    normalized_variance = smd_variance.upper().replace("-", "_")
    if normalized_variance in {"LARGE_SAMPLE", "HEDGES_1982"}:
        normalized_variance = "LS"
    if normalized_measure == "SMD" and normalized_variance != "LS":
        raise UnsupportedMethodError(
            "smd_variance currently supports only 'LS' (the Hedges 1982 "
            "large-sample approximation)."
        )

    included = studies.included
    row_count = len(included)
    effect = np.full(row_count, np.nan, dtype=np.float64)
    variance = np.full(row_count, np.nan, dtype=np.float64)
    pooled_sd = np.full(row_count, np.nan, dtype=np.float64)
    cohen_d = np.full(row_count, np.nan, dtype=np.float64)
    correction_factor = np.full(row_count, np.nan, dtype=np.float64)

    difference = studies.mean_treat - studies.mean_control
    if normalized_measure == "MD":
        effect[included] = difference[included]
        variance[included] = (
            studies.sd_treat[included] ** 2 / studies.n_treat[included]
            + studies.sd_control[included] ** 2 / studies.n_control[included]
        )
    else:
        degrees_freedom = studies.n_treat + studies.n_control - 2.0
        pooled_variance = (
            (studies.n_treat - 1.0) * studies.sd_treat**2
            + (studies.n_control - 1.0) * studies.sd_control**2
        ) / degrees_freedom
        invalid_pooled = included & (
            ~np.isfinite(pooled_variance) | (pooled_variance <= 0.0)
        )
        if np.any(invalid_pooled):
            rows = np.flatnonzero(invalid_pooled).tolist()
            raise InvalidStudyDataError(
                "SMD requires a finite, strictly positive pooled SD; invalid "
                f"rows: {rows}."
            )

        pooled_sd[included] = np.sqrt(pooled_variance[included])
        cohen_d[included] = difference[included] / pooled_sd[included]
        correction_factor[included] = _exact_hedges_correction(
            degrees_freedom[included]
        )
        effect[included] = correction_factor[included] * cohen_d[included]
        total_n = studies.n_treat + studies.n_control
        variance[included] = (
            1.0 / studies.n_treat[included]
            + 1.0 / studies.n_control[included]
            + effect[included] ** 2 / (2.0 * total_n[included])
        )

    invalid_variance = included & (~np.isfinite(variance) | (variance <= 0.0))
    if np.any(invalid_variance):
        rows = np.flatnonzero(invalid_variance).tolist()
        raise InvalidStudyDataError(
            "Continuous sampling variances must be finite and strictly positive; "
            f"invalid rows: {rows}."
        )

    return ContinuousEffectData(
        studies=studies,
        effect=effect,
        variance=variance,
        pooled_sd=pooled_sd,
        cohen_d=cohen_d,
        correction_factor=correction_factor,
        measure=normalized_measure,
    )
