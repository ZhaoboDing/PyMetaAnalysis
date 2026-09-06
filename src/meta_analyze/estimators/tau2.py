"""Between-study variance estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from ..exceptions import (
    ConvergenceError,
    InsufficientStudiesError,
    InvalidStudyDataError,
    UnsupportedMethodError,
)
from ..heterogeneity import _scaled_q_components, _weight_trace, _weighted_residuals


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
    if not np.isfinite(upper):
        raise ConvergenceError("Could not construct a finite tau-squared bracket.")
    for expansion in range(max_expansions + 1):
        try:
            value = function(upper)
        except InvalidStudyDataError as error:
            raise ConvergenceError(
                "Could not bracket a finite tau-squared solution."
            ) from error
        if np.isfinite(value) and value <= 0.0:
            return upper, expansion
        upper *= 4.0
        if not np.isfinite(upper):
            break
    raise ConvergenceError("Could not bracket a finite tau-squared solution.")


def _dersimonian_laird(
    effect: NDArray[np.float64], variance: NDArray[np.float64]
) -> Tau2Estimate:
    variance_scale = float(np.min(variance))
    relative_weights = variance_scale / variance
    q_scaled, _ = _scaled_q_components(effect, variance)
    df = len(effect) - 1
    c_scaled = _weight_trace(relative_weights)
    value = max(0.0, (q_scaled - df * variance_scale) / c_scaled)
    if not np.isfinite(value):
        raise InvalidStudyDataError(
            "DL tau-squared is not representable as a finite float."
        )
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
        with np.errstate(over="ignore", invalid="ignore"):
            denominator = variance + tau2
        if np.any(~np.isfinite(denominator)):
            raise InvalidStudyDataError("Tau-squared denominators must remain finite.")
        q_scaled, scale = _scaled_q_components(effect, denominator)
        return q_scaled - df * scale

    at_zero = equation(0.0)
    if at_zero <= 0.0:
        return Tau2Estimate(0.0, "PM", True, 0, True)

    with np.errstate(over="ignore", invalid="ignore"):
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
    value = max(0.0, float(root))
    return Tau2Estimate(
        value=value,
        method="PM",
        converged=True,
        iterations=expansions + result.iterations,
        boundary=value == 0.0,
    )


def _reml_score(
    effect: NDArray[np.float64], variance: NDArray[np.float64], tau2: float
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = variance + tau2
    if np.any(~np.isfinite(denominator)):
        raise InvalidStudyDataError("Tau-squared denominators must remain finite.")
    variance_scale = float(np.min(denominator))
    relative_weights = variance_scale / denominator
    if np.any(relative_weights == 0.0):
        raise InvalidStudyDataError(
            "Variance ratio is too large to represent all relative weights."
        )
    trace_scaled = _weight_trace(relative_weights)
    with np.errstate(over="ignore", invalid="ignore"):
        residual = _weighted_residuals(effect, relative_weights)
        # Divide the score by its positive trace before subtracting. Multiplying
        # trace by variance_scale first can underflow at extreme precision ratios.
        weighted_residual = (relative_weights / np.sqrt(trace_scaled)) * residual
        quadratic = float(np.dot(weighted_residual, weighted_residual))
    if np.isnan(quadratic):
        raise InvalidStudyDataError("REML score is not numerically defined.")
    return 0.5 * (quadratic - variance_scale)


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

    with np.errstate(over="ignore", invalid="ignore"):
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
    value = max(0.0, float(root))
    return Tau2Estimate(
        value=value,
        method="REML",
        converged=True,
        iterations=expansions + result.iterations,
        boundary=value == 0.0,
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

    try:
        effect = np.asarray(effect, dtype=np.float64)
        variance = np.asarray(variance, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidStudyDataError(
            "Effects and variances must be numeric arrays."
        ) from error
    if effect.ndim != 1 or variance.shape != effect.shape:
        raise InvalidStudyDataError(
            "Effects and variances must be aligned one-dimensional arrays."
        )
    if (
        np.any(~np.isfinite(effect))
        or np.any(~np.isfinite(variance))
        or np.any(variance <= 0)
    ):
        raise InvalidStudyDataError(
            "Effects must be finite and variances finite and positive."
        )
    if not isinstance(method, str):
        raise UnsupportedMethodError("tau2_method must be 'DL', 'PM', or 'REML'.")
    if isinstance(atol, bool) or not np.isfinite(atol) or atol <= 0:
        raise InvalidStudyDataError("atol must be finite and strictly positive.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise InvalidStudyDataError("max_iter must be a positive integer.")
    if len(effect) < 2:
        raise InsufficientStudiesError(
            "Tau-squared estimation requires at least two studies."
        )
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
