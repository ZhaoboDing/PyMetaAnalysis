"""Weighted common- and mixed-effects meta-regression estimation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import exp, fsum, log

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import chi2, f, norm, t

from ..exceptions import (
    ConvergenceError,
    InsufficientStudiesError,
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
class _PrecisionGeometry:
    denominator: NDArray[np.float64]
    variance_scale: float
    relative_weights: NDArray[np.float64]
    scaled_design: NDArray[np.float64]
    weighted_design: NDArray[np.float64]
    row_order: NDArray[np.int64]
    column_scales: NDArray[np.float64]
    orthonormal_design: NDArray[np.float64]
    upper_triangular: NDArray[np.float64]
    inverse_gram: NDArray[np.float64]
    covariance: NDArray[np.float64]


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
    q_scaled: float
    trace_p_scaled: float
    weighted_square_residual_scaled: float
    variance_scale: float


def _precision_geometry(
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
    tau2: float,
) -> _PrecisionGeometry:
    """Return the shared precision geometry for a supplied tau-squared value."""

    with np.errstate(over="ignore", invalid="ignore"):
        denominator = variance + tau2
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise InvalidStudyDataError(
            "Fitted precision denominators must be finite and positive."
        )
    variance_scale = float(np.min(denominator))
    relative_weights = variance_scale / denominator
    if np.any(relative_weights == 0.0):
        raise InvalidStudyDataError(
            "Variance ratio is too large to represent all relative weights."
        )
    column_maxima = np.max(np.abs(design_matrix), axis=0)
    if np.any(~np.isfinite(column_maxima)) or np.any(column_maxima <= 0.0):
        raise InvalidStudyDataError(
            "Meta-regression design columns must have finite positive scales."
        )
    _, column_exponents = np.frexp(column_maxima)
    column_scales = np.ldexp(np.ones_like(column_maxima), column_exponents - 1)
    scaled_design = design_matrix / column_scales
    if np.linalg.matrix_rank(scaled_design) != scaled_design.shape[1]:
        raise InvalidStudyDataError("Meta-regression design matrix is rank deficient.")
    sort_keys = tuple(
        [
            scaled_design[:, position]
            for position in reversed(range(scaled_design.shape[1]))
        ]
        + [-relative_weights]
    )
    row_order = np.lexsort(sort_keys).astype(np.int64, copy=False)
    ordered_design = scaled_design[row_order]
    ordered_weights = relative_weights[row_order]
    weighted_design = np.sqrt(ordered_weights)[:, np.newaxis] * ordered_design
    try:
        orthonormal_design, upper_triangular = np.linalg.qr(
            weighted_design, mode="reduced"
        )
        inverse_upper = np.linalg.solve(
            upper_triangular, np.eye(upper_triangular.shape[0])
        )
    except (
        np.linalg.LinAlgError
    ) as error:  # pragma: no cover - rank checked at boundary
        raise InvalidStudyDataError(
            "Meta-regression design matrix could not be solved stably."
        ) from error
    inverse_gram = inverse_upper @ inverse_upper.T
    inverse_gram = 0.5 * (inverse_gram + inverse_gram.T)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_scales = 1.0 / column_scales
        covariance = (
            variance_scale
            * inverse_gram
            * inverse_scales[:, np.newaxis]
            * inverse_scales[np.newaxis, :]
        )
    if np.any(~np.isfinite(covariance)):
        raise InvalidStudyDataError(
            "Meta-regression coefficient covariance is not representable as "
            "finite floats."
        )
    return _PrecisionGeometry(
        denominator=denominator,
        variance_scale=variance_scale,
        relative_weights=relative_weights,
        scaled_design=scaled_design,
        weighted_design=weighted_design,
        row_order=row_order,
        column_scales=column_scales,
        orthonormal_design=orthonormal_design,
        upper_triangular=upper_triangular,
        inverse_gram=inverse_gram,
        covariance=covariance,
    )


def coefficient_covariance_at_tau2(
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
    tau2: float,
) -> NDArray[np.float64]:
    """Return the classic coefficient covariance at a supplied tau-squared."""

    return _precision_geometry(variance, design_matrix, tau2).covariance


def _weighted_solution(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
    tau2: float,
) -> _WeightedSolution:
    geometry = _precision_geometry(variance, design_matrix, tau2)
    constant_columns = np.flatnonzero(
        np.all(design_matrix == design_matrix[0, :], axis=0)
        & (design_matrix[0, :] != 0.0)
    )
    anchor_position = int(constant_columns[0]) if constant_columns.size else None
    effect_anchor = (
        float(effect[geometry.row_order[0]]) if anchor_position is not None else 0.0
    )
    centered_effect = effect - effect_anchor
    try:
        ordered_centered_effect = centered_effect[geometry.row_order]
        scaled_coefficients = np.linalg.solve(
            geometry.upper_triangular,
            geometry.orthonormal_design.T
            @ (
                np.sqrt(geometry.relative_weights[geometry.row_order])
                * ordered_centered_effect
            ),
        )
    except (
        np.linalg.LinAlgError
    ) as error:  # pragma: no cover - rank checked at boundary
        raise InvalidStudyDataError(
            "Meta-regression design matrix could not be solved stably."
        ) from error
    try:
        ordered_design = geometry.scaled_design[geometry.row_order]
        ordered_weights = geometry.relative_weights[geometry.row_order]
        normal_gram = ordered_design.T @ (
            ordered_weights[:, np.newaxis] * ordered_design
        )
        normal_coefficients = np.linalg.solve(
            normal_gram,
            ordered_design.T @ (ordered_weights * ordered_centered_effect),
        )
    except np.linalg.LinAlgError:
        pass
    else:
        normal_residuals = centered_effect - (
            geometry.scaled_design @ normal_coefficients
        )
        if np.all(normal_residuals == 0.0):
            scaled_coefficients = normal_coefficients

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        coefficients = scaled_coefficients / geometry.column_scales
    if anchor_position is not None:
        coefficients[anchor_position] += (
            effect_anchor / design_matrix[0, anchor_position]
        )
    if np.any(~np.isfinite(coefficients)):
        raise InvalidStudyDataError(
            "Meta-regression coefficients are not representable as finite floats."
        )
    centered_fitted_values = geometry.scaled_design @ scaled_coefficients
    fitted_values = centered_fitted_values + effect_anchor
    residuals = centered_effect - centered_fitted_values
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        precision_weights = 1.0 / geometry.denominator
    if np.any(~np.isfinite(precision_weights)) or np.any(precision_weights <= 0.0):
        raise InvalidStudyDataError(
            "Sampling variances are too small for finite precision weights."
        )
    normalized = geometry.relative_weights / float(np.sum(geometry.relative_weights))
    ordered_leverage = np.einsum(
        "ij,ij->i",
        geometry.orthonormal_design,
        geometry.orthonormal_design,
    )
    leverage = np.empty_like(ordered_leverage)
    leverage[geometry.row_order] = np.clip(ordered_leverage, 0.0, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_residuals = np.sqrt(geometry.relative_weights) * residuals
        q_scaled = float(np.dot(scaled_residuals, scaled_residuals))
        q = float(q_scaled / geometry.variance_scale)
    if np.isnan(q):
        raise InvalidStudyDataError(
            "The weighted residual sum of squares is undefined."
        )

    trace_relative_p = _stable_projection_trace(
        geometry.relative_weights[geometry.row_order],
        geometry.scaled_design[geometry.row_order],
        geometry.weighted_design,
        geometry.upper_triangular,
        np.clip(ordered_leverage, 0.0, 1.0),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_residuals = geometry.relative_weights * residuals
        weighted_square_residual_scaled = float(
            np.dot(weighted_residuals, weighted_residuals)
        )
    return _WeightedSolution(
        coefficients=coefficients,
        covariance=geometry.covariance,
        fitted_values=fitted_values,
        residuals=residuals,
        precision_weights=precision_weights,
        normalized_precision_weights=normalized,
        leverage=leverage,
        q=q,
        q_scaled=q_scaled,
        trace_p_scaled=trace_relative_p,
        weighted_square_residual_scaled=weighted_square_residual_scaled,
        variance_scale=geometry.variance_scale,
    )


def _stable_projection_trace(
    relative_weights: NDArray[np.float64],
    scaled_design: NDArray[np.float64],
    weighted_design: NDArray[np.float64],
    upper_triangular: NDArray[np.float64],
    leverage: NDArray[np.float64],
) -> float:
    """Return ``sum(w_i * (1-h_i))`` without global cancellation."""

    complements = 1.0 - leverage
    high_leverage = np.flatnonzero(leverage > 0.5)
    if high_leverage.size:
        diagonal = np.abs(np.diag(upper_triangular))
        if np.any(diagonal <= 0.0):
            raise InvalidStudyDataError(
                "Meta-regression precision geometry is not positive definite."
            )
        log_determinant = 2.0 * fsum(log(value) for value in diagonal)
        for position in high_leverage:
            retained = np.arange(len(relative_weights)) != position
            retained_design = weighted_design[retained]
            if (
                np.linalg.matrix_rank(scaled_design[retained])
                != retained_design.shape[1]
            ):
                complements[position] = 0.0
                continue
            _, retained_upper = np.linalg.qr(retained_design, mode="reduced")
            retained_diagonal = np.abs(np.diag(retained_upper))
            if np.any(retained_diagonal <= 0.0):
                complements[position] = 0.0
                continue
            retained_log_determinant = 2.0 * fsum(
                log(value) for value in retained_diagonal
            )
            log_complement = retained_log_determinant - log_determinant
            complements[position] = (
                0.0
                if log_complement < log(np.finfo(np.float64).tiny)
                else min(1.0, exp(log_complement))
            )
    trace = fsum((relative_weights * np.clip(complements, 0.0, 1.0)).tolist())
    if not np.isfinite(trace) or trace <= 0.0:
        raise InvalidStudyDataError(
            "Precision ratios are too extreme to retain positive residual information."
        )
    return trace


def _find_upper_bound(
    function: Callable[[float], float],
    *,
    initial: float,
    max_expansions: int,
) -> tuple[float, int]:
    upper = max(float(initial), np.finfo(np.float64).tiny)
    if not np.isfinite(upper):
        raise ConvergenceError(
            "Could not construct a finite meta-regression tau-squared bracket."
        )
    for expansion in range(max_expansions + 1):
        try:
            value = function(upper)
        except InvalidStudyDataError as error:
            raise ConvergenceError(
                "Could not bracket a finite meta-regression tau-squared solution."
            ) from error
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
    if residual_df <= 0:
        raise InsufficientStudiesError(
            "Meta-regression tau-squared estimation requires positive residual "
            f"degrees of freedom; got k={len(effect)} and "
            f"p={design_matrix.shape[1]}."
        )
    at_zero = _weighted_solution(effect, variance, design_matrix, 0.0)

    if normalized_method == "DL":
        value = max(
            0.0,
            (at_zero.q_scaled - residual_df * at_zero.variance_scale)
            / at_zero.trace_p_scaled,
        )
        if not np.isfinite(value):
            raise InvalidStudyDataError(
                "DL meta-regression tau-squared is not representable as a finite float."
            )
        return Tau2Estimate(value, "DL", True, 0, value == 0.0)

    if normalized_method == "PM":

        def equation(tau2: float) -> float:
            solution = _weighted_solution(effect, variance, design_matrix, tau2)
            return float(solution.q_scaled - residual_df * solution.variance_scale)

    else:

        def equation(tau2: float) -> float:
            solution = _weighted_solution(effect, variance, design_matrix, tau2)
            quadratic = (
                solution.weighted_square_residual_scaled / solution.trace_p_scaled
            )
            return 0.5 * (quadratic - solution.variance_scale)

    at_boundary = equation(0.0)
    if at_boundary <= 0.0:
        return Tau2Estimate(0.0, normalized_method, True, 0, True)

    with np.errstate(over="ignore", invalid="ignore"):
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
    value = max(0.0, float(root))
    return Tau2Estimate(
        value=value,
        method=normalized_method,
        converged=True,
        iterations=expansions + result.iterations,
        boundary=value == 0.0,
    )


def residual_heterogeneity(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Return residual QE plus scaled trace and variance components."""

    solution = _weighted_solution(effect, variance, design_matrix, 0.0)
    return solution.q, solution.trace_p_scaled, solution.variance_scale


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
        out=np.full_like(coefficients, np.nan),
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
    wald = _wald_statistic(selected, selected_covariance)
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


def _wald_statistic(
    estimates: NDArray[np.float64], covariance: NDArray[np.float64]
) -> float:
    """Return a Wald statistic, unavailable for exactly zero HK covariance."""
    if np.all(covariance == 0.0):
        return float("nan")
    try:
        value = float(estimates @ np.linalg.solve(covariance, estimates))
    except np.linalg.LinAlgError as error:
        raise InvalidStudyDataError(
            "Wald test covariance could not be solved."
        ) from error
    return max(0.0, value)


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
                "weighted residual variation; joint Wald tests are unavailable."
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
        out=np.full_like(solution.coefficients, np.nan),
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
