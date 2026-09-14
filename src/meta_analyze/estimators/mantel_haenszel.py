"""Mantel-Haenszel common-effect estimators for binary outcomes.

The pooled OR/RR estimates and Greenland-Robins variance equations follow the
publicly documented Review Manager 5 statistical algorithms. Risk differences
use the Sato-Greenland-Robins variance, which remains consistent under both
large-stratum and sparse-data limiting models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import norm

from ..exceptions import (
    InsufficientStudiesError,
    InvalidStudyDataError,
    UnsupportedMethodError,
)


@dataclass(frozen=True, slots=True)
class MantelHaenszelFit:
    """Numerical outputs from a Mantel-Haenszel common-effect model."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    weights: NDArray[np.float64]
    normalized_weights: NDArray[np.float64]


def fit_mantel_haenszel(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
    *,
    measure: str,
    confidence_level: float,
) -> MantelHaenszelFit:
    """Fit a Mantel-Haenszel common-effect log OR, log RR, or RD."""

    normalized_measure = measure.upper()
    if normalized_measure not in {"OR", "RR", "RD"}:
        raise UnsupportedMethodError(
            "Mantel-Haenszel supports measure='OR', measure='RR', or measure='RD'."
        )

    tables = (a, b, c, d)
    if (
        any(table.ndim != 1 for table in tables)
        or len({len(table) for table in tables}) != 1
    ):
        raise InvalidStudyDataError(
            "Mantel-Haenszel cell arrays must be one-dimensional and have equal "
            "lengths."
        )
    if len(a) == 0:
        raise InsufficientStudiesError(
            "Mantel-Haenszel pooling requires at least one study."
        )
    cells = np.concatenate(tables)
    if np.any(~np.isfinite(cells)) or np.any(cells < 0.0):
        raise InvalidStudyDataError(
            "Mantel-Haenszel cell counts must be finite and non-negative."
        )
    zero_total = (a == 0.0) & (b == 0.0) & (c == 0.0) & (d == 0.0)
    if np.any(zero_total):
        rows = np.flatnonzero(zero_total).tolist()
        raise InvalidStudyDataError(
            f"Mantel-Haenszel strata have zero total at row positions {rows}."
        )

    cell_scale = float(np.max(cells))
    row_scale = np.maximum.reduce(tables)
    relative_scale = row_scale / cell_scale
    with np.errstate(under="ignore"):
        scaled_a = a / row_scale
        scaled_b = b / row_scale
        scaled_c = c / row_scale
        scaled_d = d / row_scale
    total = scaled_a + scaled_b + scaled_c + scaled_d
    n1 = scaled_a + scaled_b
    n2 = scaled_c + scaled_d
    with np.errstate(divide="ignore"):
        log_cell_scale = float(np.log(cell_scale))
        log_relative_scale = np.log(row_scale) - log_cell_scale
        log_a = np.log(scaled_a)
        log_b = np.log(scaled_b)
        log_c = np.log(scaled_c)
        log_d = np.log(scaled_d)
        log_total = np.log(total)
        log_n1 = np.log(n1)
        log_n2 = np.log(n2)
    if normalized_measure == "OR":
        log_r = float(logsumexp(log_relative_scale + log_a + log_d - log_total))
        log_s = float(logsumexp(log_relative_scale + log_b + log_c - log_total))
        if not np.isfinite(log_r) or not np.isfinite(log_s):
            raise InvalidStudyDataError(
                "The exact Mantel-Haenszel OR is undefined because its pooled "
                "cross-product is zero; set a positive mh_continuity_correction."
            )

        with np.errstate(divide="ignore"):
            log_ad_sum = np.log(scaled_a + scaled_d)
            log_bc_sum = np.log(scaled_b + scaled_c)
        log_e = float(
            logsumexp(log_relative_scale + log_ad_sum + log_a + log_d - 2.0 * log_total)
        )
        log_f = float(
            logsumexp(log_relative_scale + log_ad_sum + log_b + log_c - 2.0 * log_total)
        )
        log_g = float(
            logsumexp(log_relative_scale + log_bc_sum + log_a + log_d - 2.0 * log_total)
        )
        log_h = float(
            logsumexp(log_relative_scale + log_bc_sum + log_b + log_c - 2.0 * log_total)
        )
        log_variance = float(
            np.log(0.5)
            + logsumexp(
                [
                    log_e - 2.0 * log_r,
                    np.logaddexp(log_f, log_g) - log_r - log_s,
                    log_h - 2.0 * log_s,
                ]
            )
            - log_cell_scale
        )
        with np.errstate(over="ignore", under="ignore"):
            pooled_variance = float(np.exp(log_variance))
        estimate = log_r - log_s
        log_weights = log_relative_scale + log_b + log_c - log_total
        weights = scaled_b * scaled_c / total * row_scale
    elif normalized_measure == "RR":
        log_r = float(logsumexp(log_relative_scale + log_a + log_n2 - log_total))
        log_s = float(logsumexp(log_relative_scale + log_c + log_n1 - log_total))
        if not np.isfinite(log_r) or not np.isfinite(log_s):
            raise InvalidStudyDataError(
                "The exact Mantel-Haenszel RR is undefined because the pooled "
                "event total is zero; set a positive mh_continuity_correction."
            )

        # Algebraically positive expansion of
        # n1*n2*(a+c) - a*c*(n1+n2), avoiding cancellation.
        log_p = float(
            logsumexp(
                np.concatenate(
                    (
                        log_relative_scale + log_a + log_d + log_n1 - 2.0 * log_total,
                        log_relative_scale + log_b + log_c + log_n2 - 2.0 * log_total,
                    )
                )
            )
        )
        log_variance = log_p - log_r - log_s - log_cell_scale
        with np.errstate(over="ignore", under="ignore"):
            pooled_variance = float(np.exp(log_variance))
        estimate = log_r - log_s
        log_weights = log_relative_scale + log_c + log_n1 - log_total
        weights = scaled_c * n1 / total * row_scale
    else:
        # The MH risk difference is the arm-size-weighted mean of the raw
        # study risk differences. All arithmetic below uses counts divided by
        # their common maximum; the Sato-Greenland-Robins variance is therefore
        # divided by cell_scale once to restore the original count scale.
        treat_fraction = n1 / total
        control_fraction = n2 / total
        local_a = scaled_a
        local_c = scaled_c
        scaled_a = local_a * relative_scale
        scaled_c = local_c * relative_scale
        scaled_n1 = n1 * relative_scale
        scaled_n2 = n2 * relative_scale
        scaled_weights = scaled_n1 * control_fraction
        weight_sum = float(np.sum(scaled_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise InvalidStudyDataError(
                "Mantel-Haenszel RD study weights have a non-positive sum."
            )

        pooled = float(
            np.sum(scaled_a * control_fraction - scaled_c * treat_fraction) / weight_sum
        )
        linear_component = float(
            np.sum(
                scaled_c * treat_fraction**2
                - scaled_a * control_fraction**2
                + treat_fraction * control_fraction * (scaled_n2 - scaled_n1) / 2.0
            )
        )
        binomial_component = float(
            np.sum(
                relative_scale
                * (local_a * (n2 - local_c) / total + local_c * (n1 - local_a) / total)
            )
            / 2.0
        )
        pooled_variance = (
            (pooled * linear_component + binomial_component)
            / weight_sum**2
            / cell_scale
        )
        estimate = pooled

    if normalized_measure != "RD":
        log_weight_sum = float(logsumexp(log_weights))
        with np.errstate(under="ignore"):
            normalized_weights = np.exp(log_weights - log_weight_sum)
    else:
        weight_sum = float(np.sum(scaled_weights))
        weights = scaled_weights * cell_scale
        normalized_weights = scaled_weights / weight_sum

    if (
        not np.isfinite(estimate)
        or not np.isfinite(pooled_variance)
        or pooled_variance <= 0.0
    ):
        if normalized_measure == "RD":
            raise InvalidStudyDataError(
                "Mantel-Haenszel RD produced a non-positive sampling variance "
                "under the Sato-Greenland-Robins method. PyMetaAnalysis rejects "
                "this degenerate case instead of returning a zero or near-zero "
                "standard error; set a positive "
                "mh_continuity_correction if the review protocol permits it."
            )
        raise InvalidStudyDataError(
            "Mantel-Haenszel produced a non-positive sampling variance."
        )
    if np.any(~np.isfinite(weights)) or not np.isfinite(
        float(np.sum(normalized_weights))
    ):
        raise InvalidStudyDataError(
            "Mantel-Haenszel study weights have a non-positive sum."
        )

    standard_error = float(np.sqrt(pooled_variance))
    critical_value = float(norm.ppf(0.5 + float(confidence_level) / 2.0))
    margin = critical_value * standard_error
    return MantelHaenszelFit(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - margin,
        ci_high=estimate + margin,
        weights=np.asarray(weights, dtype=np.float64),
        normalized_weights=np.asarray(normalized_weights, dtype=np.float64),
    )
