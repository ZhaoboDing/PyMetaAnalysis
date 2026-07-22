"""Weighted common- and mixed-effects meta-regression estimation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import chi2, f, norm, t

from ..exceptions import (
    ConvergenceError,
    InvalidStudyDataError,
    UnsupportedMethodError,
)
from .tau2 import Tau2Estimate


@dataclass(frozen=True, slots=True)
class RegressionTestFit:
    """Numerical output for a Wald test."""

    statistic: float
    statistic_name: str
    distribution: str
    df_num: int
    df_denom: int | None
    pvalue: float


@dataclass(frozen=True, slots=True)
class MetaRegressionFit:
    """Numerical outputs from a weighted meta-regression."""

    coefficients: NDArray[np.float64]
    covariance: NDArray[np.float64]
    classic_covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    ci_low: NDArray[np.float64]
    ci_high: NDArray[np.float64]
    statistics: NDArray[np.float64]
    pvalues: NDArray[np.float64]
    statistic_name: str
    coefficient_df: int | None
    global_test: RegressionTestFit
    fitted_values: NDArray[np.float64]
    residuals: NDArray[np.float64]
    precision_weights: NDArray[np.float64]
    normalized_precision_weights: NDArray[np.float64]
    leverage: NDArray[np.float64]
    tau2: Tau2Estimate | None
    residual_scale: float
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _WeightedSolution:
    coefficients: NDArray[np.float64]
    covariance: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    residuals: NDArray[np.float64]
    precision_weights: NDArray[np.float64]
    normalized_precision_weights: NDArray[np.float64]
    leverage: NDArray[np.float64]
    q: float
    trace_p: float
    weighted_square_residual: float


def _weighted_solution(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
    tau2: float,
) -> _WeightedSolution:
    denominator = variance + tau2
    variance_scale = float(np.min(denominator))
    relative_weights = variance_scale / denominator
    weighted_design = relative_weights[:, np.newaxis] * design_matrix
    gram = design_matrix.T @ weighted_design
    try:
        inverse_gram = np.linalg.solve(gram, np.eye(gram.shape[0]))
        coefficients = np.linalg.solve(
            gram, design_matrix.T @ (relative_weights * effect)
        )
    except (
        np.linalg.LinAlgError
    ) as error:  # pragma: no cover - rank checked at boundary
        raise InvalidStudyDataError(
            "Meta-regression design matrix could not be solved stably."
        ) from error

    inverse_gram = 0.5 * (inverse_gram + inverse_gram.T)
    covariance = variance_scale * inverse_gram
    fitted_values = design_matrix @ coefficients
    residuals = effect - fitted_values
    precision_weights = 1.0 / denominator
    normalized = relative_weights / float(np.sum(relative_weights))
    leverage = relative_weights * np.einsum(
        "ij,jk,ik->i", design_matrix, inverse_gram, design_matrix
    )
    q = float(np.dot(precision_weights, residuals * residuals))

    squared_weight_crossproduct = design_matrix.T @ (
        (relative_weights * relative_weights)[:, np.newaxis] * design_matrix
    )
    trace_relative_p = float(
        np.sum(relative_weights) - np.trace(inverse_gram @ squared_weight_crossproduct)
    )
    trace_p = trace_relative_p / variance_scale
    weighted_square_residual = float(
        np.dot(precision_weights * precision_weights, residuals * residuals)
    )
    return _WeightedSolution(
        coefficients=coefficients,
        covariance=covariance,
        fitted_values=fitted_values,
        residuals=residuals,
        precision_weights=precision_weights,
        normalized_precision_weights=normalized,
        leverage=leverage,
        q=q,
        trace_p=trace_p,
        weighted_square_residual=weighted_square_residual,
    )


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
    raise ConvergenceError(
        "Could not bracket a finite meta-regression tau-squared solution."
    )


def estimate_meta_regression_tau2(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
    *,
    method: str,
    atol: float,
    max_iter: int,
) -> Tau2Estimate:
    """Estimate residual tau-squared for a full-rank meta-regression design."""

    normalized_method = method.upper().replace("-", "_")
    if normalized_method not in {"DL", "PM", "REML"}:
        raise UnsupportedMethodError(
            f"Unsupported tau2_method={method!r}; expected 'DL', 'PM', or 'REML'."
        )
    residual_df = len(effect) - design_matrix.shape[1]
    at_zero = _weighted_solution(effect, variance, design_matrix, 0.0)

    if normalized_method == "DL":
        value = max(0.0, (at_zero.q - residual_df) / at_zero.trace_p)
        return Tau2Estimate(value, "DL", True, 0, value == 0.0)

    if normalized_method == "PM":

        def equation(tau2: float) -> float:
            return float(
                _weighted_solution(effect, variance, design_matrix, tau2).q
                - residual_df
            )

    else:

        def equation(tau2: float) -> float:
            solution = _weighted_solution(effect, variance, design_matrix, tau2)
            return 0.5 * (solution.weighted_square_residual - solution.trace_p)

    at_boundary = equation(0.0)
    if at_boundary <= 0.0:
        return Tau2Estimate(0.0, normalized_method, True, 0, True)

    sample_variance = float(np.var(effect, ddof=1))
    initial = max(sample_variance, float(np.max(variance)))
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
        raise ConvergenceError(
            f"{normalized_method} meta-regression tau-squared estimation failed."
        ) from error
    if not result.converged:
        raise ConvergenceError(
            f"{normalized_method} meta-regression tau-squared estimation did "
            "not converge."
        )
    return Tau2Estimate(
        value=max(0.0, float(root)),
        method=normalized_method,
        converged=True,
        iterations=expansions + result.iterations,
        boundary=root <= atol,
    )


def residual_heterogeneity(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
) -> tuple[float, float]:
    """Return residual QE and the trace of its common-effect P matrix."""

    solution = _weighted_solution(effect, variance, design_matrix, 0.0)
    return solution.q, solution.trace_p


def _coefficient_statistics(
    coefficients: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    confidence_level: float,
    distribution: str,
    residual_df: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    str,
    int | None,
]:
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.zeros_like(coefficients),
        where=standard_errors > 0.0,
    )
    nonzero_with_zero_se = (standard_errors == 0.0) & (coefficients != 0.0)
    statistics[nonzero_with_zero_se] = (
        np.sign(coefficients[nonzero_with_zero_se]) * np.inf
    )
    alpha = 1.0 - confidence_level
    if distribution == "normal":
        critical = float(norm.ppf(1.0 - alpha / 2.0))
        pvalues = 2.0 * norm.sf(np.abs(statistics))
        statistic_name = "z"
        coefficient_df = None
    else:
        critical = float(t.ppf(1.0 - alpha / 2.0, df=residual_df))
        pvalues = 2.0 * t.sf(np.abs(statistics), df=residual_df)
        statistic_name = "t"
        coefficient_df = residual_df
    margin = critical * standard_errors
    return (
        standard_errors,
        coefficients - margin,
        coefficients + margin,
        np.asarray(pvalues, dtype=np.float64),
        statistic_name,
        coefficient_df,
    )


def _global_test(
    coefficients: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    intercept: bool,
    inference_method: str,
    residual_df: int,
) -> RegressionTestFit:
    start = 1 if intercept else 0
    selected = coefficients[start:]
    selected_covariance = covariance[start:, start:]
    term_count = len(selected)
    if term_count == 0:
        return RegressionTestFit(
            statistic=0.0,
            statistic_name="not_applicable",
            distribution="not_applicable",
            df_num=0,
            df_denom=None,
            pvalue=float("nan"),
        )
    try:
        wald = float(selected @ np.linalg.solve(selected_covariance, selected))
    except np.linalg.LinAlgError as error:  # pragma: no cover - full rank checked
        raise InvalidStudyDataError(
            "Global moderator test could not be solved."
        ) from error
    wald = max(0.0, wald)
    if inference_method == "normal":
        return RegressionTestFit(
            statistic=wald,
            statistic_name="chi_square",
            distribution="chi_square",
            df_num=term_count,
            df_denom=None,
            pvalue=float(chi2.sf(wald, df=term_count)),
        )
    statistic = wald / term_count
    return RegressionTestFit(
        statistic=statistic,
        statistic_name="F",
        distribution="F",
        df_num=term_count,
        df_denom=residual_df,
        pvalue=float(f.sf(statistic, term_count, residual_df)),
    )


def fit_meta_regression(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
    *,
    intercept: bool,
    model: str,
    tau2_method: str,
    inference_method: str,
    confidence_level: float,
    atol: float,
    max_iter: int,
) -> MetaRegressionFit:
    """Fit a common- or mixed-effects inverse-variance meta-regression."""

    if model not in {"common", "mixed"}:
        raise UnsupportedMethodError("model must be 'common' or 'mixed'.")
    if inference_method not in {
        "normal",
        "hartung_knapp",
        "hartung_knapp_adhoc",
    }:
        raise UnsupportedMethodError(
            "inference_method must be 'normal', 'hartung_knapp', or "
            "'hartung_knapp_adhoc'."
        )
    if model == "common" and inference_method != "normal":
        raise UnsupportedMethodError(
            "Hartung-Knapp inference is only supported for mixed-effects models."
        )
    normalized_tau2 = tau2_method.upper().replace("-", "_")
    if normalized_tau2 not in {"DL", "PM", "REML"}:
        raise UnsupportedMethodError(
            f"Unsupported tau2_method={tau2_method!r}; expected 'DL', 'PM', or 'REML'."
        )

    residual_df = len(effect) - design_matrix.shape[1]
    tau2: Tau2Estimate | None
    if model == "common":
        tau2 = None
        tau2_value = 0.0
    else:
        tau2 = estimate_meta_regression_tau2(
            effect,
            variance,
            design_matrix,
            method=normalized_tau2,
            atol=atol,
            max_iter=max_iter,
        )
        tau2_value = tau2.value

    solution = _weighted_solution(effect, variance, design_matrix, tau2_value)
    classic_covariance = solution.covariance
    residual_scale = 1.0
    warnings: list[str] = []
    if inference_method == "normal":
        covariance = classic_covariance
        distribution = "normal"
    else:
        residual_scale = max(0.0, solution.q / residual_df)
        hk_covariance = residual_scale * classic_covariance
        if inference_method == "hartung_knapp_adhoc":
            residual_scale = max(1.0, residual_scale)
            covariance = residual_scale * classic_covariance
        else:
            covariance = hk_covariance
            if residual_scale < 1.0:
                warnings.append(
                    "Hartung-Knapp produced coefficient variances below the classic "
                    "variances; use inference_method='hartung_knapp_adhoc' for "
                    "lower-bound protection."
                )
        if residual_scale == 0.0:
            warnings.append(
                "Hartung-Knapp variance is zero because the fitted model has no "
                "weighted residual variation."
            )
        distribution = "t"

    (
        standard_errors,
        ci_low,
        ci_high,
        pvalues,
        statistic_name,
        coefficient_df,
    ) = _coefficient_statistics(
        solution.coefficients,
        covariance,
        confidence_level=confidence_level,
        distribution=distribution,
        residual_df=residual_df,
    )
    statistics = np.divide(
        solution.coefficients,
        standard_errors,
        out=np.zeros_like(solution.coefficients),
        where=standard_errors > 0.0,
    )
    nonzero_with_zero_se = (standard_errors == 0.0) & (solution.coefficients != 0.0)
    statistics[nonzero_with_zero_se] = (
        np.sign(solution.coefficients[nonzero_with_zero_se]) * np.inf
    )

    return MetaRegressionFit(
        coefficients=solution.coefficients,
        covariance=covariance,
        classic_covariance=classic_covariance,
        standard_errors=standard_errors,
        ci_low=ci_low,
        ci_high=ci_high,
        statistics=statistics,
        pvalues=pvalues,
        statistic_name=statistic_name,
        coefficient_df=coefficient_df,
        global_test=_global_test(
            solution.coefficients,
            covariance,
            intercept=intercept,
            inference_method=inference_method,
            residual_df=residual_df,
        ),
        fitted_values=solution.fitted_values,
        residuals=solution.residuals,
        precision_weights=solution.precision_weights,
        normalized_precision_weights=solution.normalized_precision_weights,
        leverage=solution.leverage,
        tau2=tau2,
        residual_scale=residual_scale,
        warnings=tuple(warnings),
    )
