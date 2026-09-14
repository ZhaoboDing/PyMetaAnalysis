"""Peto one-step common-effect odds-ratio estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2, norm

from ..exceptions import InsufficientStudiesError, InvalidStudyDataError


@dataclass(frozen=True, slots=True)
class PetoFit:
    """Numerical outputs from a Peto common-effect odds-ratio model."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    weights: NDArray[np.float64]
    normalized_weights: NDArray[np.float64]
    q: float
    q_df: int
    q_pvalue: float
    i2: float
    h2: float


def fit_peto(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
    *,
    confidence_level: float,
) -> PetoFit:
    """Fit Peto's one-step common-effect log odds ratio.

    Every supplied stratum must have positive hypergeometric information, so
    callers must first exclude double-zero and double-all tables. The public
    ``meta_binary`` API performs that filtering. This differs from the pooling
    side of ``metafor::rma.peto()`` under its default ``drop00`` handling,
    which can retain zero-information strata in ``k`` and Q degrees of freedom.
    """

    tables = (a, b, c, d)
    if (
        any(table.ndim != 1 for table in tables)
        or len({len(table) for table in tables}) != 1
    ):
        raise InvalidStudyDataError(
            "Peto cell arrays must be one-dimensional and have equal lengths."
        )
    if len(a) == 0:
        raise InsufficientStudiesError("Peto pooling requires at least one study.")
    cells = np.concatenate(tables)
    if np.any(~np.isfinite(cells)) or np.any(cells < 0.0):
        raise InvalidStudyDataError("Peto cell counts must be finite and non-negative.")

    cell_scale = float(np.max(cells))
    if cell_scale <= 0.0:
        raise InvalidStudyDataError(
            "Peto pooling requires at least one observed outcome."
        )
    invalid_arms = ((a == 0.0) & (b == 0.0)) | ((c == 0.0) & (d == 0.0))
    if np.any(invalid_arms):
        rows = np.flatnonzero(invalid_arms).tolist()
        raise InvalidStudyDataError(
            f"Peto strata must have two positive group totals; invalid rows: {rows}."
        )
    row_scale = np.maximum.reduce(tables)
    with np.errstate(under="ignore"):
        scaled_a = a / row_scale
        scaled_b = b / row_scale
        scaled_c = c / row_scale
        scaled_d = d / row_scale
    n1 = scaled_a + scaled_b
    n0 = scaled_c + scaled_d
    total = n1 + n0
    with np.errstate(divide="ignore", over="ignore"):
        total_minus_one = total - 1.0 / row_scale
    invalid_total = total_minus_one <= 0.0
    if np.any(invalid_total):
        rows = np.flatnonzero(invalid_total).tolist()
        raise InvalidStudyDataError(
            f"Peto strata require total sample size above one; invalid rows: {rows}."
        )

    outcome_events = scaled_a + scaled_c
    outcome_nonevents = scaled_b + scaled_d
    observed_minus_expected = (scaled_a * n0 - scaled_c * n1) / total
    local_information = (
        outcome_events
        * outcome_nonevents
        * (n1 * n0 / (total * total))
        / total_minus_one
    )
    invalid_information = ~np.isfinite(local_information) | (local_information <= 0.0)
    if np.any(invalid_information):
        rows = np.flatnonzero(invalid_information).tolist()
        raise InvalidStudyDataError(
            "Peto pooling requires both outcomes to occur within every included "
            f"stratum; invalid rows: {rows}."
        )

    with np.errstate(divide="ignore"):
        log_cell_scale = float(np.log(cell_scale))
        log_relative_scale = np.log(row_scale) - log_cell_scale
        log_information = log_relative_scale + np.log(local_information)
        log_information_sum = float(logsumexp(log_information))
        log_abs_oe = log_relative_scale + np.log(np.abs(observed_minus_expected))
    log_abs_oe_sum, oe_sum_sign = logsumexp(
        log_abs_oe,
        b=np.sign(observed_minus_expected),
        return_sign=True,
    )
    if not np.isfinite(log_information_sum):
        raise InvalidStudyDataError(
            "Peto hypergeometric information has a non-positive sum."
        )

    with np.errstate(over="ignore", under="ignore"):
        estimate = (
            0.0
            if oe_sum_sign == 0.0
            else float(oe_sum_sign * np.exp(log_abs_oe_sum - log_information_sum))
        )
        pooled_variance = float(np.exp(-log_information_sum - log_cell_scale))
    if not np.isfinite(estimate) or not np.isfinite(pooled_variance):
        raise InvalidStudyDataError(
            "Peto's pooled estimate or sampling variance is not finite."
        )
    if pooled_variance <= 0.0:
        raise InvalidStudyDataError("Peto produced a non-positive sampling variance.")

    standard_error = float(np.sqrt(pooled_variance))
    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    margin = critical_value * standard_error
    with np.errstate(over="ignore", invalid="ignore"):
        residual = observed_minus_expected - estimate * local_information
    with np.errstate(divide="ignore"):
        log_q_terms = (
            log_relative_scale
            + 2.0 * np.log(np.abs(residual))
            - np.log(local_information)
        )
    with np.errstate(over="ignore"):
        q = float(np.exp(logsumexp(log_q_terms) + log_cell_scale))
    if np.isnan(q):
        raise InvalidStudyDataError("Peto's heterogeneity statistic is undefined.")

    q_df = len(a) - 1
    if q_df == 0:
        q = 0.0
        q_pvalue = float("nan")
        i2 = float("nan")
        h2 = float("nan")
    elif np.isinf(q):
        q_pvalue = 0.0
        i2 = 1.0
        h2 = float("inf")
    else:
        q_pvalue = float(chi2.sf(q, q_df))
        i2 = 0.0 if q <= 0.0 else max(0.0, (q - q_df) / q)
        h2 = q / q_df

    with np.errstate(over="ignore", under="ignore"):
        weights = local_information * row_scale
        normalized_weights = np.exp(log_information - log_information_sum)
    if np.any(~np.isfinite(weights)):
        raise InvalidStudyDataError("Peto study weights are not finite.")
    return PetoFit(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - margin,
        ci_high=estimate + margin,
        weights=weights,
        normalized_weights=normalized_weights,
        q=q,
        q_df=q_df,
        q_pvalue=q_pvalue,
        i2=i2,
        h2=h2,
    )
