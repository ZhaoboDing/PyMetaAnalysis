"""Mantel-Haenszel common-effect estimators for binary outcomes.

The pooled estimates and Greenland-Robins variance equations follow the
publicly documented Review Manager 5 statistical algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..exceptions import InvalidStudyDataError, UnsupportedMethodError


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
    """Fit a Mantel-Haenszel common-effect log OR or log RR."""

    normalized_measure = measure.upper()
    if normalized_measure not in {"OR", "RR"}:
        raise UnsupportedMethodError(
            "Mantel-Haenszel currently supports measure='OR' or measure='RR'."
        )

    total = a + b + c + d
    n1 = a + b
    n2 = c + d
    if normalized_measure == "OR":
        r = float(np.sum(a * d / total))
        s = float(np.sum(b * c / total))
        if r <= 0.0 or s <= 0.0:
            raise InvalidStudyDataError(
                "The exact Mantel-Haenszel OR is undefined because its pooled "
                "cross-product is zero; set a positive mh_continuity_correction."
            )

        e = float(np.sum((a + d) * a * d / total**2))
        f = float(np.sum((a + d) * b * c / total**2))
        g = float(np.sum((b + c) * a * d / total**2))
        h = float(np.sum((b + c) * b * c / total**2))
        pooled = r / s
        pooled_variance = 0.5 * (e / r**2 + (f + g) / (r * s) + h / s**2)
        weights = b * c / total
    else:
        r = float(np.sum(a * n2 / total))
        s = float(np.sum(c * n1 / total))
        if r <= 0.0 or s <= 0.0:
            raise InvalidStudyDataError(
                "The exact Mantel-Haenszel RR is undefined because the pooled "
                "event total is zero; set a positive mh_continuity_correction."
            )

        p = float(np.sum((n1 * n2 * (a + c) - a * c * total) / total**2))
        pooled = r / s
        pooled_variance = p / (r * s)
        weights = c * n1 / total

    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        raise InvalidStudyDataError(
            "Mantel-Haenszel produced a non-positive sampling variance."
        )
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise InvalidStudyDataError(
            "Mantel-Haenszel study weights have a non-positive sum."
        )

    estimate = float(np.log(pooled))
    standard_error = float(np.sqrt(pooled_variance))
    critical_value = float(norm.ppf(0.5 + float(confidence_level) / 2.0))
    margin = critical_value * standard_error
    return MantelHaenszelFit(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - margin,
        ci_high=estimate + margin,
        weights=weights,
        normalized_weights=weights / weight_sum,
    )
