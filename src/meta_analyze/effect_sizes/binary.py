"""Effect sizes and validation for two-group binary outcomes.

OR, RR, and RD equations follow the publicly documented Review Manager 5
statistical algorithms by Deeks and Higgins (2010).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypeAlias, cast

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

CorrectionScope: TypeAlias = Literal[
    "only_zero_studies", "all_studies", "if_any_zero", "none"
]
RDZeroVariancePolicy: TypeAlias = Literal["correct", "exclude"]


@dataclass(frozen=True, slots=True)
class BinaryStudies:
    """Validated raw counts and row-level inclusion metadata."""

    row_id: NDArray[np.int64]
    study: NDArray[np.object_]
    event_treat: NDArray[np.float64]
    n_treat: NDArray[np.float64]
    event_control: NDArray[np.float64]
    n_control: NDArray[np.float64]
    included: NDArray[np.bool_]
    exclusion_reason: NDArray[np.object_]


@dataclass(frozen=True, slots=True)
class BinaryEffectData:
    """Binary study effects on the model scale."""

    studies: BinaryStudies
    effect: NDArray[np.float64]
    variance: NDArray[np.float64]
    corrected: NDArray[np.bool_]
    rd_zero_variance: NDArray[np.bool_]
    measure: str
    effect_scale: str
    display_scale: str

    @property
    def included_effect(self) -> NDArray[np.float64]:
        return self.effect[self.studies.included]

    @property
    def included_variance(self) -> NDArray[np.float64]:
        return self.variance[self.studies.included]


def normalize_correction_scope(scope: str) -> CorrectionScope:
    normalized = scope.lower().replace("-", "_")
    aliases = {
        "only0": "only_zero_studies",
        "only_zero_cells": "only_zero_studies",
        "all": "all_studies",
        "if0all": "if_any_zero",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"only_zero_studies", "all_studies", "if_any_zero", "none"}
    if normalized not in allowed:
        raise UnsupportedMethodError(
            "correction_scope must be 'only_zero_studies', 'all_studies', "
            "'if_any_zero', or 'none'."
        )
    return cast(CorrectionScope, normalized)


def normalize_rd_zero_variance(policy: str) -> RDZeroVariancePolicy:
    """Normalize the policy for RD studies with zero raw sampling variance."""

    if not isinstance(policy, str):
        raise UnsupportedMethodError("rd_zero_variance must be 'correct' or 'exclude'.")
    normalized = policy.lower().replace("-", "_")
    if normalized not in {"correct", "exclude"}:
        raise UnsupportedMethodError("rd_zero_variance must be 'correct' or 'exclude'.")
    return cast(RDZeroVariancePolicy, normalized)


def validate_correction(value: float | None, *, name: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidStudyDataError(f"{name} must be a non-negative number or None.")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise InvalidStudyDataError(f"{name} must be finite and non-negative.")
    return numeric


def _missing_reason(names: list[str]) -> str:
    return "missing " + ", ".join(names)


def normalize_binary_studies(
    *,
    data: pd.DataFrame | None,
    event_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    event_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None,
    missing: MissingPolicy,
) -> BinaryStudies:
    """Resolve and validate two-group binary study counts."""

    if data is not None and not isinstance(data, pd.DataFrame):
        raise InvalidStudyDataError("data must be a pandas DataFrame or None.")
    if missing not in {"raise", "drop"}:
        raise InvalidStudyDataError("missing must be either 'raise' or 'drop'.")

    arguments = {
        "event_treat": event_treat,
        "n_treat": n_treat,
        "event_control": event_control,
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
            f"Binary count inputs must have equal lengths; {detail}."
        )
    length = lengths.pop()
    if data is not None and len(data) != length:
        raise InvalidStudyDataError(
            "Array-like binary inputs used with data must have exactly one value "
            "per DataFrame row."
        )
    if length == 0:
        raise InvalidStudyDataError("At least one study row is required.")
    labels = _study_labels(study, data=data, length=length)

    try:
        values = {
            name: np.asarray(vector, dtype=np.float64) for name, vector in raw.items()
        }
    except (TypeError, ValueError) as error:
        raise InvalidStudyDataError(
            "Binary counts must contain numeric values."
        ) from error

    missing_by_name = {name: pd.isna(vector) for name, vector in values.items()}
    any_missing = np.logical_or.reduce(tuple(missing_by_name.values()))
    if np.any(any_missing) and missing == "raise":
        rows = np.flatnonzero(any_missing).tolist()
        raise InvalidStudyDataError(
            f"Missing binary counts at row positions {rows}; use missing='drop' "
            "to exclude them explicitly."
        )

    for name, vector in values.items():
        present = ~missing_by_name[name]
        invalid_finite = present & ~np.isfinite(vector)
        if np.any(invalid_finite):
            rows = np.flatnonzero(invalid_finite).tolist()
            raise InvalidStudyDataError(f"{name} must be finite; invalid rows: {rows}.")
        noninteger = present & (vector != np.floor(vector))
        if np.any(noninteger):
            rows = np.flatnonzero(noninteger).tolist()
            raise InvalidStudyDataError(
                f"{name} must contain whole-number counts; invalid rows: {rows}."
            )

    active = ~any_missing
    for total_name in ("n_treat", "n_control"):
        invalid = active & (values[total_name] <= 0.0)
        if np.any(invalid):
            rows = np.flatnonzero(invalid).tolist()
            raise InvalidStudyDataError(
                f"{total_name} must be strictly positive; invalid rows: {rows}."
            )
    for event_name, total_name in (
        ("event_treat", "n_treat"),
        ("event_control", "n_control"),
    ):
        invalid = active & (
            (values[event_name] < 0.0) | (values[event_name] > values[total_name])
        )
        if np.any(invalid):
            rows = np.flatnonzero(invalid).tolist()
            raise InvalidStudyDataError(
                f"{event_name} must be between 0 and {total_name}; "
                f"invalid rows: {rows}."
            )

    reasons = np.full(length, None, dtype=object)
    for index in np.flatnonzero(any_missing):
        names = [name for name, mask in missing_by_name.items() if mask[index]]
        reasons[index] = _missing_reason(names)
    if not np.any(active):
        raise InvalidStudyDataError(
            "No studies remain after applying the missing-value policy."
        )

    return BinaryStudies(
        row_id=np.arange(length, dtype=np.int64),
        study=labels,
        event_treat=values["event_treat"],
        n_treat=values["n_treat"],
        event_control=values["event_control"],
        n_control=values["n_control"],
        included=active,
        exclusion_reason=reasons,
    )


def correction_mask(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
    *,
    included: NDArray[np.bool_],
    scope: CorrectionScope,
) -> NDArray[np.bool_]:
    zero_study: NDArray[np.bool_] = np.asarray(
        (a == 0.0) | (b == 0.0) | (c == 0.0) | (d == 0.0),
        dtype=np.bool_,
    )
    if scope == "only_zero_studies":
        return included & zero_study
    if scope == "all_studies":
        return included.copy()
    if scope == "if_any_zero":
        return (
            included.copy()
            if np.any(included & zero_study)
            else np.zeros_like(included)
        )
    return np.zeros_like(included)


def adjusted_tables(
    studies: BinaryStudies,
    *,
    correction: float,
    scope: CorrectionScope,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    a = studies.event_treat.copy()
    b = studies.n_treat - studies.event_treat
    c = studies.event_control.copy()
    d = studies.n_control - studies.event_control
    corrected = correction_mask(a, b, c, d, included=studies.included, scope=scope) & (
        correction > 0.0
    )
    if np.any(corrected):
        for cell in (a, b, c, d):
            cell[corrected] += correction
    return a, b, c, d, corrected


def calculate_binary_effects(
    studies: BinaryStudies,
    *,
    measure: str,
    continuity_correction: float,
    correction_scope: CorrectionScope,
    rd_zero_variance: str = "correct",
) -> BinaryEffectData:
    """Calculate OR, RR, or RD effects and sampling variances."""

    normalized_measure = measure.upper()
    if normalized_measure not in {"OR", "RR", "RD"}:
        raise UnsupportedMethodError("measure must be 'OR', 'RR', or 'RD'.")
    rd_policy = normalize_rd_zero_variance(rd_zero_variance)
    if normalized_measure != "RD" and rd_policy != "correct":
        raise UnsupportedMethodError(
            "rd_zero_variance is only configurable when measure='RD'."
        )

    included = studies.included.copy()
    reasons = studies.exclusion_reason.copy()
    rd_zero_variance_mask = np.zeros_like(included)
    if normalized_measure == "RD":
        treat_risk = studies.event_treat / studies.n_treat
        control_risk = studies.event_control / studies.n_control
        raw_rd_variance = (
            treat_risk * (1.0 - treat_risk) / studies.n_treat
            + control_risk * (1.0 - control_risk) / studies.n_control
        )
        rd_zero_variance_mask = np.asarray(
            included & (raw_rd_variance == 0.0), dtype=np.bool_
        )
        if rd_policy == "exclude":
            included[rd_zero_variance_mask] = False
            for index in np.flatnonzero(rd_zero_variance_mask):
                reasons[index] = "zero uncorrected risk-difference variance"

    if normalized_measure in {"OR", "RR"}:
        double_zero = (studies.event_treat == 0.0) & (studies.event_control == 0.0)
        double_all = (studies.event_treat == studies.n_treat) & (
            studies.event_control == studies.n_control
        )
        uninformative = included & (double_zero | double_all)
        included[uninformative] = False
        for index in np.flatnonzero(uninformative):
            reasons[index] = (
                "no events in either group"
                if double_zero[index]
                else "all participants have events in both groups"
            )

    working_studies = BinaryStudies(
        row_id=studies.row_id,
        study=studies.study,
        event_treat=studies.event_treat,
        n_treat=studies.n_treat,
        event_control=studies.event_control,
        n_control=studies.n_control,
        included=included,
        exclusion_reason=reasons,
    )
    if not np.any(included):
        raise InvalidStudyDataError(
            f"No informative studies remain for measure={normalized_measure!r}."
        )

    a, b, c, d, corrected = adjusted_tables(
        working_studies,
        correction=continuity_correction,
        scope=correction_scope,
    )
    if normalized_measure == "OR":
        undefined = included & ((a == 0.0) | (b == 0.0) | (c == 0.0) | (d == 0.0))
    elif normalized_measure == "RR":
        # Log risk ratios require a positive event count in each arm. A zero
        # non-event cell is valid unless both arms are all-event, which was
        # excluded above as uninformative.
        undefined = included & ((a == 0.0) | (c == 0.0))
    else:
        undefined = np.zeros_like(included)
    if np.any(undefined):
        rows = np.flatnonzero(undefined).tolist()
        correction_guidance = (
            "set correction_scope to a value other than 'none' and use a positive "
            "continuity_correction"
            if correction_scope == "none"
            else "use a positive continuity_correction"
        )
        raise InvalidStudyDataError(
            f"{normalized_measure} study effects are undefined with remaining zero "
            f"cells at row positions {rows}; {correction_guidance}."
        )

    effect = np.full(len(included), np.nan, dtype=np.float64)
    variance = np.full(len(included), np.nan, dtype=np.float64)
    active = included
    if normalized_measure == "OR":
        effect[active] = (
            np.log(a[active])
            + np.log(d[active])
            - np.log(b[active])
            - np.log(c[active])
        )
        variance[active] = (
            1.0 / a[active] + 1.0 / b[active] + 1.0 / c[active] + 1.0 / d[active]
        )
        effect_scale = "log"
        display_scale = "exp"
    elif normalized_measure == "RR":
        n1 = a + b
        n2 = c + d
        effect[active] = np.log((a[active] / n1[active]) / (c[active] / n2[active]))
        variance[active] = (
            1.0 / a[active] - 1.0 / n1[active] + 1.0 / c[active] - 1.0 / n2[active]
        )
        effect_scale = "log"
        display_scale = "exp"
    else:
        # RD remains on the uncorrected natural scale. Its variance uses corrected
        # counts for every study selected by correction_scope; under the default
        # scope this means every table containing at least one zero cell.
        effect[active] = (
            studies.event_treat[active] / studies.n_treat[active]
            - studies.event_control[active] / studies.n_control[active]
        )
        n1 = a + b
        n2 = c + d
        treat_risk = a[active] / n1[active]
        control_risk = c[active] / n2[active]
        variance[active] = (
            treat_risk * (1.0 - treat_risk) / n1[active]
            + control_risk * (1.0 - control_risk) / n2[active]
        )
        effect_scale = "identity"
        display_scale = "identity"

    invalid_variance = active & (~np.isfinite(variance) | (variance <= 0.0))
    if np.any(invalid_variance):
        rows = np.flatnonzero(invalid_variance).tolist()
        correction_guidance = (
            "set correction_scope to a value other than 'none' and use a positive "
            "continuity_correction"
            if correction_scope == "none"
            else "use an appropriate positive continuity_correction"
        )
        if normalized_measure == "RD":
            correction_guidance += " or set rd_zero_variance='exclude'"
        raise InvalidStudyDataError(
            f"Non-positive binary sampling variance at row positions {rows}; "
            f"{correction_guidance}."
        )
    invalid_effect = active & ~np.isfinite(effect)
    if np.any(invalid_effect):
        rows = np.flatnonzero(invalid_effect).tolist()
        raise InvalidStudyDataError(
            "Binary effect sizes must be finite after calculation; "
            f"invalid rows: {rows}."
        )
    _validate_finite_precision_variance(
        variance,
        included=active,
        label="Binary sampling variances",
    )

    return BinaryEffectData(
        studies=working_studies,
        effect=effect,
        variance=variance,
        corrected=corrected,
        rd_zero_variance=rd_zero_variance_mask,
        measure=normalized_measure,
        effect_scale=effect_scale,
        display_scale=display_scale,
    )


def calculate_peto_effects(
    studies: BinaryStudies,
    *,
    continuity_correction: float,
    correction_scope: CorrectionScope,
) -> BinaryEffectData:
    """Calculate study-level Peto log odds ratios and sampling variances."""

    included = studies.included.copy()
    reasons = studies.exclusion_reason.copy()
    double_zero = (studies.event_treat == 0.0) & (studies.event_control == 0.0)
    double_all = (studies.event_treat == studies.n_treat) & (
        studies.event_control == studies.n_control
    )
    uninformative = included & (double_zero | double_all)
    included[uninformative] = False
    for index in np.flatnonzero(uninformative):
        reasons[index] = (
            "no events in either group"
            if double_zero[index]
            else "all participants have events in both groups"
        )
    if not np.any(included):
        raise InvalidStudyDataError("No informative studies remain for Peto OR.")

    working_studies = BinaryStudies(
        row_id=studies.row_id,
        study=studies.study,
        event_treat=studies.event_treat,
        n_treat=studies.n_treat,
        event_control=studies.event_control,
        n_control=studies.n_control,
        included=included,
        exclusion_reason=reasons,
    )
    a, b, c, d, corrected = adjusted_tables(
        working_studies,
        correction=continuity_correction,
        scope=correction_scope,
    )

    effect = np.full(len(included), np.nan, dtype=np.float64)
    variance = np.full(len(included), np.nan, dtype=np.float64)
    active_a = a[included]
    active_b = b[included]
    active_c = c[included]
    active_d = d[included]
    cell_scale = np.maximum.reduce((active_a, active_b, active_c, active_d))
    scaled_a = active_a / cell_scale
    scaled_b = active_b / cell_scale
    scaled_c = active_c / cell_scale
    scaled_d = active_d / cell_scale
    n1 = scaled_a + scaled_b
    n0 = scaled_c + scaled_d
    total = n1 + n0
    outcome_events = scaled_a + scaled_c
    outcome_nonevents = scaled_b + scaled_d
    observed_minus_expected = (scaled_a * n0 - scaled_c * n1) / total
    information = (
        outcome_events
        * outcome_nonevents
        * (n1 * n0 / (total * total))
        / (total - 1.0 / cell_scale)
    )
    effect[included] = observed_minus_expected / information
    variance[included] = (1.0 / information) / cell_scale

    invalid_effect = included & ~np.isfinite(effect)
    invalid_variance = included & (~np.isfinite(variance) | (variance <= 0.0))
    if np.any(invalid_effect | invalid_variance):
        rows = np.flatnonzero(invalid_effect | invalid_variance).tolist()
        raise InvalidStudyDataError(
            "Peto study effects require positive hypergeometric information; "
            f"invalid rows: {rows}."
        )
    _validate_finite_precision_variance(
        variance,
        included=included,
        label="Peto sampling variances",
    )

    return BinaryEffectData(
        studies=working_studies,
        effect=effect,
        variance=variance,
        corrected=corrected,
        rd_zero_variance=np.zeros_like(included),
        measure="OR",
        effect_scale="log",
        display_scale="exp",
    )
