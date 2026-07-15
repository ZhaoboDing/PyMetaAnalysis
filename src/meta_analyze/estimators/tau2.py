"""Between-study variance estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from ..exceptions import ConvergenceError, UnsupportedMethodError
from ..heterogeneity import generalized_q, weighted_mean


@dataclass(frozen=True, slots=True)
class Tau2Estimate:
    """A between-study variance estimate and convergence metadata."""

    value: float
    method: str
    converged: bool
    iterations: int
    boundary: bool


def _find_upper_bound(
    function: Callable[[float], float],
    *,
    initial: float,
    max_expansions: int,
) -> tuple[float, int]:
    upper = max(float(initial), np.finfo(np.float64).tiny)
    for expansion in range(max_expansions + 1):
        value = function(upper)
        if np.isfinite(value) and value <= 0.0:
            return upper, expansion
        upper *= 4.0
        if not np.isfinite(upper):
            break
    raise ConvergenceError("Could not bracket a finite tau-squared solution.")


def _dersimonian_laird(
    effect: NDArray[np.float64], variance: NDArray[np.float64]
) -> Tau2Estimate:
    weights = 1.0 / variance
    estimate = weighted_mean(effect, weights)
    residual = effect - estimate
    q = float(np.dot(weights, residual * residual))
    df = len(effect) - 1
    weight_sum = float(np.sum(weights))
    c = weight_sum - float(np.dot(weights, weights)) / weight_sum
    value = max(0.0, (q - df) / c)
    return Tau2Estimate(
        value=value,
        method="DL",
        converged=True,
        iterations=0,
        boundary=value == 0.0,
    )


def _paule_mandel(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    *,
    atol: float,
    max_iter: int,
) -> Tau2Estimate:
    df = len(effect) - 1

    def equation(tau2: float) -> float:
        return generalized_q(effect, variance, tau2) - df

    at_zero = equation(0.0)
    if at_zero <= 0.0:
        return Tau2Estimate(0.0, "PM", True, 0, True)

    initial = max(float(np.var(effect, ddof=1)), float(np.max(variance)))
    upper, expansions = _find_upper_bound(
        equation, initial=initial, max_expansions=max_iter
    )
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
        raise ConvergenceError("Paule-Mandel tau-squared estimation failed.") from error
    if not result.converged:
        raise ConvergenceError("Paule-Mandel tau-squared estimation did not converge.")
    return Tau2Estimate(
        value=max(0.0, float(root)),
        method="PM",
        converged=True,
        iterations=expansions + result.iterations,
        boundary=root <= atol,
    )


def _reml_score(
    effect: NDArray[np.float64], variance: NDArray[np.float64], tau2: float
) -> float:
    weights = 1.0 / (variance + tau2)
    estimate = weighted_mean(effect, weights)
    residual = effect - estimate
    weighted_square_residual = float(np.dot(weights * weights, residual * residual))
    trace_p = float(np.sum(weights) - np.dot(weights, weights) / np.sum(weights))
    return 0.5 * (weighted_square_residual - trace_p)


def _restricted_maximum_likelihood(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    *,
    atol: float,
    max_iter: int,
) -> Tau2Estimate:
    def score(tau2: float) -> float:
        return _reml_score(effect, variance, tau2)

    at_zero = score(0.0)
    if at_zero <= 0.0:
        return Tau2Estimate(0.0, "REML", True, 0, True)

    initial = max(float(np.var(effect, ddof=1)), float(np.max(variance)))
    upper, expansions = _find_upper_bound(
        score, initial=initial, max_expansions=max_iter
    )
    try:
        root, result = brentq(
            score,
            0.0,
            upper,
            xtol=atol,
            rtol=max(atol, 4.0 * np.finfo(np.float64).eps),
            maxiter=max_iter,
            full_output=True,
            disp=False,
        )
    except (RuntimeError, ValueError) as error:
        raise ConvergenceError("REML tau-squared estimation failed.") from error
    if not result.converged:
        raise ConvergenceError("REML tau-squared estimation did not converge.")
    return Tau2Estimate(
        value=max(0.0, float(root)),
        method="REML",
        converged=True,
        iterations=expansions + result.iterations,
        boundary=root <= atol,
    )


def estimate_tau2(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    *,
    method: str,
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> Tau2Estimate:
    """Estimate between-study variance using DL, PM, or REML."""

    normalized_method = method.upper().replace("-", "_")
    if normalized_method == "DL":
        return _dersimonian_laird(effect, variance)
    if normalized_method == "PM":
        return _paule_mandel(effect, variance, atol=atol, max_iter=max_iter)
    if normalized_method == "REML":
        return _restricted_maximum_likelihood(
            effect, variance, atol=atol, max_iter=max_iter
        )
    raise UnsupportedMethodError(
        f"Unsupported tau2_method={method!r}; expected 'DL', 'PM', or 'REML'."
    )
