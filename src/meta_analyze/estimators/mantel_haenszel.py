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

    cell_scale = float(np.max(np.concatenate((a, b, c, d))))
    scaled_a = a / cell_scale
    scaled_b = b / cell_scale
    scaled_c = c / cell_scale
    scaled_d = d / cell_scale
    total = scaled_a + scaled_b + scaled_c + scaled_d
    n1 = scaled_a + scaled_b
    n2 = scaled_c + scaled_d
    if normalized_measure == "OR":
        r = float(np.sum(scaled_a * scaled_d / total))
        s = float(np.sum(scaled_b * scaled_c / total))
        if r <= 0.0 or s <= 0.0:
            raise InvalidStudyDataError(
                "The exact Mantel-Haenszel OR is undefined because its pooled "
                "cross-product is zero; set a positive mh_continuity_correction."
            )

        e = float(np.sum((scaled_a + scaled_d) * scaled_a * scaled_d / total**2))
        f = float(np.sum((scaled_a + scaled_d) * scaled_b * scaled_c / total**2))
        g = float(np.sum((scaled_b + scaled_c) * scaled_a * scaled_d / total**2))
        h = float(np.sum((scaled_b + scaled_c) * scaled_b * scaled_c / total**2))
        pooled = r / s
        pooled_variance = 0.5 * (e / r**2 + (f + g) / (r * s) + h / s**2) / cell_scale
        scaled_weights = scaled_b * scaled_c / total
    else:
        r = float(np.sum(scaled_a * n2 / total))
        s = float(np.sum(scaled_c * n1 / total))
        if r <= 0.0 or s <= 0.0:
            raise InvalidStudyDataError(
                "The exact Mantel-Haenszel RR is undefined because the pooled "
                "event total is zero; set a positive mh_continuity_correction."
            )

        p = float(
            np.sum(
                (n1 * n2 * (scaled_a + scaled_c) - scaled_a * scaled_c * total)
                / total**2
            )
        )
        pooled = r / s
        pooled_variance = p / (r * s) / cell_scale
        scaled_weights = scaled_c * n1 / total

    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        raise InvalidStudyDataError(
            "Mantel-Haenszel produced a non-positive sampling variance."
        )
    weight_sum = float(np.sum(scaled_weights))
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
        weights=scaled_weights * cell_scale,
        normalized_weights=scaled_weights / weight_sum,
    )
