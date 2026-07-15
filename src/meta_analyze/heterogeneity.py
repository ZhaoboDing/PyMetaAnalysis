"""Classical heterogeneity statistics for univariate meta-analysis."""

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
