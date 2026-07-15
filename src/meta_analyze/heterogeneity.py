"""Heterogeneity statistics for univariate meta-analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2


def weighted_mean(effect: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Return a weighted mean without overflowing the raw weight sum."""

    largest = float(np.max(weights))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("Weights must contain a finite, strictly positive value.")
    scaled = weights / largest
    scaled_sum = float(np.sum(scaled))
    if not np.isfinite(scaled_sum) or scaled_sum <= 0.0:
        raise ValueError("Weights must have a finite, strictly positive sum.")
    if bool(np.all(effect == effect[0])):
        return float(effect[0])
    return float(np.dot(scaled, effect) / scaled_sum)


def generalized_q(
    effect: NDArray[np.float64], variance: NDArray[np.float64], tau2: float
) -> float:
    """Return the weighted residual Q statistic at a given tau-squared."""

    weights = 1.0 / (variance + tau2)
    estimate = weighted_mean(effect, weights)
    residual = effect - estimate
    return float(np.dot(weights, residual * residual))


def classical_heterogeneity(
    effect: NDArray[np.float64], variance: NDArray[np.float64]
) -> tuple[float, int, float, float, float]:
    """Return Q, degrees of freedom, p-value, I-squared, and H-squared."""

    k = len(effect)
    if k == 1:
        return 0.0, 0, float("nan"), float("nan"), float("nan")

    weights = 1.0 / variance
    estimate = weighted_mean(effect, weights)
    return heterogeneity_at_estimate(effect, variance, estimate)


def heterogeneity_at_estimate(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    estimate: float,
) -> tuple[float, int, float, float, float]:
    """Return heterogeneity statistics around an explicitly pooled estimate."""

    k = len(effect)
    if k == 1:
        return 0.0, 0, float("nan"), float("nan"), float("nan")

    weights = 1.0 / variance
    residual = effect - estimate
    q = float(np.dot(weights, residual * residual))
    df = k - 1
    pvalue = float(chi2.sf(q, df))
    i2 = 0.0 if q <= 0.0 else max(0.0, (q - df) / q)
    h2 = q / df
    return q, df, pvalue, i2, h2


def tau2_inconsistency(
    variance: NDArray[np.float64], tau2: float
) -> tuple[float, float]:
    """Return tau-squared-based I-squared and H-squared.

    The typical within-study variance is ``(k - 1) / C``, where ``C`` is
    calculated from common-effect inverse-variance weights.  Scaled weights
    keep the calculation stable when sampling variances are very small.
    """

    k = len(variance)
    if k == 1:
        return float("nan"), float("nan")
    if tau2 == 0.0:
        return 0.0, 1.0

    variance_scale = float(np.min(variance))
    relative_weights = variance_scale / variance
    weight_sum = float(np.sum(relative_weights))
    c_scaled = (
        weight_sum - float(np.dot(relative_weights, relative_weights)) / weight_sum
    )
    if not np.isfinite(c_scaled) or c_scaled <= 0.0:
        return float("nan"), float("nan")

    typical_variance = (k - 1) * variance_scale / c_scaled
    i2 = 1.0 / (1.0 + typical_variance / tau2)
    h2 = 1.0 + tau2 / typical_variance
    return float(i2), float(h2)
