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
        if not np.isfinite(r) or not np.isfinite(s) or r <= 0.0 or s <= 0.0:
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
    elif normalized_measure == "RR":
        r = float(np.sum(scaled_a * n2 / total))
        s = float(np.sum(scaled_c * n1 / total))
        if not np.isfinite(r) or not np.isfinite(s) or r <= 0.0 or s <= 0.0:
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
    else:
        # The MH risk difference is the arm-size-weighted mean of the raw
        # study risk differences. All arithmetic below uses counts divided by
        # their common maximum; the Sato-Greenland-Robins variance is therefore
        # divided by cell_scale once to restore the original count scale.
        treat_fraction = n1 / total
        control_fraction = n2 / total
        scaled_weights = n1 * control_fraction
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
                + treat_fraction * control_fraction * (n2 - n1) / 2.0
            )
        )
        binomial_component = float(
            np.sum(
                scaled_a * (n2 - scaled_c) / total + scaled_c * (n1 - scaled_a) / total
            )
            / 2.0
        )
        pooled_variance = (
            (pooled * linear_component + binomial_component)
            / weight_sum**2
            / cell_scale
        )

    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        if normalized_measure == "RD":
            raise InvalidStudyDataError(
                "Mantel-Haenszel RD produced a non-positive sampling variance "
                "under the Sato-Greenland-Robins method; set a positive "
                "mh_continuity_correction if the review protocol permits it."
            )
        raise InvalidStudyDataError(
            "Mantel-Haenszel produced a non-positive sampling variance."
        )
    weight_sum = float(np.sum(scaled_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise InvalidStudyDataError(
            "Mantel-Haenszel study weights have a non-positive sum."
        )

    estimate = pooled if normalized_measure == "RD" else float(np.log(pooled))
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
