"""Heterogeneity statistics for univariate meta-analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import chi2

from .exceptions import ConvergenceError, InvalidStudyDataError


@dataclass(frozen=True, slots=True)
class QProfileInterval:
    """Internal Q-profile tau-squared bounds and numerical metadata."""

    ci_low: float
    ci_high: float
    is_empty: bool
    iterations: int


def weighted_mean(effect: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Return a weighted mean without overflowing the raw weight sum."""

    largest = float(np.max(weights))
    if not np.isfinite(largest) or largest <= 0.0:
        raise InvalidStudyDataError(
            "Weights must contain a finite, strictly positive value."
        )
    scaled = weights / largest
    scaled_sum = float(np.sum(scaled))
    if not np.isfinite(scaled_sum) or scaled_sum <= 0.0:
        raise InvalidStudyDataError(
            "Weights must have a finite, strictly positive sum."
        )
    if bool(np.all(effect == effect[0])):
        return float(effect[0])
    normalized = scaled / scaled_sum
    with np.errstate(over="ignore", invalid="ignore"):
        estimate = float(np.dot(normalized, effect))
    if not np.isfinite(estimate):
        raise InvalidStudyDataError(
            "The weighted mean is not representable as a finite float."
        )
    return estimate


def _weighted_residuals(
    effect: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Center before averaging so a dominant study retains its small residual."""
    with np.errstate(over="ignore", invalid="ignore"):
        offset = effect - effect[int(np.argmax(weights))]
        if np.all(np.isfinite(offset)):
            return np.asarray(offset - weighted_mean(offset, weights), dtype=np.float64)
        return np.asarray(effect - weighted_mean(effect, weights), dtype=np.float64)


def _weight_trace(weights: NDArray[np.float64]) -> float:
    """Compute S1 - S2/S1 as positive pair products in linear time.

    The usual subtraction loses all information when one precision dominates.
    A reverse cumulative sum supplies sum(w[j], j > i) without subtracting a
    dominant weight from the total.
    """
    tail_sums = np.cumsum(weights[:0:-1])[::-1]
    trace = float(2.0 * np.dot(weights[:-1], tail_sums) / np.sum(weights))
    if not np.isfinite(trace) or trace <= 0.0:
        raise InvalidStudyDataError(
            "Variance ratio is too large to resolve between-study precision."
        )
    return trace


def _scaled_q_components(
    effect: NDArray[np.float64],
    denominator: NDArray[np.float64],
    *,
    estimate: float | None = None,
) -> tuple[float, float]:
    """Return ``Q * scale`` and the positive denominator scale."""

    scale = float(np.min(denominator))
    relative_weights = scale / denominator
    if np.any(relative_weights == 0.0):
        raise InvalidStudyDataError(
            "Variance ratio is too large to represent all relative weights."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        residual = (
            _weighted_residuals(effect, relative_weights)
            if estimate is None
            else effect - estimate
        )
        weighted_residual = np.sqrt(relative_weights) * residual
        numerator = float(np.dot(weighted_residual, weighted_residual))
    if np.isnan(numerator):
        raise InvalidStudyDataError(
            "The weighted residual sum of squares is undefined."
        )
    return numerator, scale


def _unscale_nonnegative(value: float, scale: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(value / scale)
    if np.isnan(result):
        raise InvalidStudyDataError("The weighted statistic is undefined.")
    return result


def generalized_q(
    effect: NDArray[np.float64], variance: NDArray[np.float64], tau2: float
) -> float:
    """Return the weighted residual Q statistic at a given tau-squared."""

    numerator, scale = _scaled_q_components(effect, variance + tau2)
    return _unscale_nonnegative(numerator, scale)


def _q_profile_equation(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    tau2: float,
    target: float,
) -> float:
    """Return a scaled equation with the same root as ``Q(tau2) - target``."""

    with np.errstate(over="ignore", invalid="ignore"):
        denominator = variance + tau2
    if np.any(~np.isfinite(denominator)):
        raise InvalidStudyDataError("Q-profile denominators must remain finite.")
    numerator, scale = _scaled_q_components(effect, denominator)
    return float(numerator - target * scale)


def _q_profile_upper_bracket(
    equation: Callable[[float], float],
    *,
    initial: float,
    max_expansions: int,
) -> tuple[float, int]:
    upper = max(float(initial), np.finfo(np.float64).tiny)
    for expansion in range(max_expansions + 1):
        try:
            value = equation(upper)
        except ValueError as error:
            raise ConvergenceError(
                "Could not bracket a finite Q-profile tau-squared confidence bound."
            ) from error
        if np.isfinite(value) and value <= 0.0:
            return upper, expansion
        upper *= 4.0
        if not np.isfinite(upper):
            break
    raise ConvergenceError(
        "Could not bracket a finite Q-profile tau-squared confidence bound."
    )


def _solve_q_profile_bound(
    equation: Callable[[float], float],
    *,
    upper: float,
    atol: float,
    max_iter: int,
    label: str,
) -> tuple[float, int]:
    try:
        root, result = brentq(
            equation,
            0.0,
            upper,
            xtol=atol,
            rtol=max(atol, 4.0 * np.finfo(np.float64).eps),
            maxiter=max_iter,
            full_output=True,
            disp=False,
        )
    except (RuntimeError, ValueError) as error:
        raise ConvergenceError(
            f"Q-profile {label} tau-squared confidence bound failed."
        ) from error
    if not result.converged:
        raise ConvergenceError(
            f"Q-profile {label} tau-squared confidence bound did not converge."
        )
    return max(0.0, float(root)), result.iterations


def q_profile_tau2_interval(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    *,
    confidence_level: float,
    tau2_estimate: float,
    atol: float,
    max_iter: int,
) -> QProfileInterval:
    """Invert pooled-mean Q to obtain a constrained Q-profile interval.

    This implementation is intentionally limited to an intercept-only pooled
    effect, whose Q degrees of freedom are ``k - 1``. Meta-regression requires
    refitting the full design matrix at each candidate tau-squared in addition
    to using ``k - p`` degrees of freedom and must not reuse this function by
    changing the degrees of freedom alone.
    """

    df = len(effect) - 1
    alpha = 1.0 - confidence_level
    lower_bound_target = float(chi2.ppf(1.0 - alpha / 2.0, df))
    upper_bound_target = float(chi2.ppf(alpha / 2.0, df))

    def lower_equation(tau2: float) -> float:
        return _q_profile_equation(effect, variance, tau2, lower_bound_target)

    def upper_equation(tau2: float) -> float:
        return _q_profile_equation(effect, variance, tau2, upper_bound_target)

    lower_at_zero = lower_equation(0.0)
    upper_at_zero = upper_equation(0.0)

    if upper_at_zero < 0.0:
        return QProfileInterval(0.0, 0.0, True, 0)
    if upper_at_zero == 0.0:
        return QProfileInterval(0.0, 0.0, False, 0)

    initial = max(tau2_estimate, float(np.max(variance)))
    upper_bracket, expansions = _q_profile_upper_bracket(
        upper_equation,
        initial=initial,
        max_expansions=max_iter,
    )
    iterations = expansions
    if lower_at_zero <= 0.0:
        ci_low = 0.0
    else:
        ci_low, lower_iterations = _solve_q_profile_bound(
            lower_equation,
            upper=upper_bracket,
            atol=atol,
            max_iter=max_iter,
            label="lower",
        )
        iterations += lower_iterations

    ci_high, upper_iterations = _solve_q_profile_bound(
        upper_equation,
        upper=upper_bracket,
        atol=atol,
        max_iter=max_iter,
        label="upper",
    )
    iterations += upper_iterations
    return QProfileInterval(ci_low, ci_high, False, iterations)


def classical_heterogeneity(
    effect: NDArray[np.float64], variance: NDArray[np.float64]
) -> tuple[float, int, float, float, float]:
    """Return Q, degrees of freedom, p-value, I-squared, and H-squared."""

    k = len(effect)
    if k == 1:
        return 0.0, 0, float("nan"), float("nan"), float("nan")

    numerator, scale = _scaled_q_components(effect, variance)
    q = _unscale_nonnegative(numerator, scale)
    return _heterogeneity_from_q(q, k)


def _heterogeneity_from_q(q: float, k: int) -> tuple[float, int, float, float, float]:
    df = k - 1
    if np.isinf(q):
        return q, df, 0.0, 1.0, float("inf")
    pvalue = float(chi2.sf(q, df))
    i2 = 0.0 if q <= 0.0 else max(0.0, (q - df) / q)
    h2 = q / df
    return q, df, pvalue, i2, h2


def heterogeneity_at_estimate(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    estimate: float,
) -> tuple[float, int, float, float, float]:
    """Return heterogeneity statistics around an explicitly pooled estimate."""

    k = len(effect)
    if k == 1:
        return 0.0, 0, float("nan"), float("nan"), float("nan")

    numerator, scale = _scaled_q_components(
        effect,
        variance,
        estimate=estimate,
    )
    q = _unscale_nonnegative(numerator, scale)
    return _heterogeneity_from_q(q, k)


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
    c_scaled = _weight_trace(relative_weights)

    typical_variance = (k - 1) * variance_scale / c_scaled
    i2 = 1.0 / (1.0 + typical_variance / tau2)
    h2 = 1.0 + tau2 / typical_variance
    return float(i2), float(h2)
