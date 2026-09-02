"""Regression diagnostics for funnel-plot asymmetry and small-study effects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t

from .exceptions import InsufficientStudiesError, InvalidStudyDataError

if TYPE_CHECKING:
    from .results import MetaAnalysisResult


@dataclass(frozen=True, slots=True)
class EggerTestResult:
    """Classical Egger regression test for funnel-plot asymmetry."""

    intercept: float
    intercept_standard_error: float
    intercept_ci_low: float
    intercept_ci_high: float
    statistic: float
    statistic_name: str
    distribution: str
    df: int
    pvalue: float
    limit_estimate: float
    limit_standard_error: float
    limit_ci_low: float
    limit_ci_high: float
    confidence_level: float
    k: int
    measure: str
    effect_scale: str
    display_scale: str
    method: str
    model: str
    predictor: str
    condition_number: float
    warnings: tuple[str, ...]

    @property
    def intercept_ci(self) -> tuple[float, float]:
        """Return the confidence interval for the asymmetry intercept."""

        return self.intercept_ci_low, self.intercept_ci_high

    @property
    def limit_ci(self) -> tuple[float, float]:
        """Return the model-scale confidence interval for the limit estimate."""

        return self.limit_ci_low, self.limit_ci_high

    def _to_display_scale(self, value: float) -> float:
        if self.display_scale == "identity":
            return value
        if self.display_scale == "exp":
            return math.exp(value)
        if self.display_scale == "tanh":
            return math.tanh(value)
        raise ValueError(f"Unknown display scale {self.display_scale!r}.")

    @property
    def display_limit_estimate(self) -> float:
        """Return the limit estimate on the analysis display scale."""

        return self._to_display_scale(self.limit_estimate)

    @property
    def display_limit_ci(self) -> tuple[float, float]:
        """Return the limit-estimate interval on the display scale."""

        return (
            self._to_display_scale(self.limit_ci_low),
            self._to_display_scale(self.limit_ci_high),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, machine-readable representation."""

        return {
            "method": self.method,
            "model": self.model,
            "predictor": self.predictor,
            "studies": self.k,
            "measure": self.measure,
            "effect_scale": self.effect_scale,
            "display_scale": self.display_scale,
            "confidence_level": self.confidence_level,
            "intercept": self.intercept,
            "intercept_standard_error": self.intercept_standard_error,
            "intercept_ci_low": self.intercept_ci_low,
            "intercept_ci_high": self.intercept_ci_high,
            "statistic": self.statistic,
            "statistic_name": self.statistic_name,
            "distribution": self.distribution,
            "df": self.df,
            "pvalue": self.pvalue,
            "limit_estimate": self.limit_estimate,
            "limit_standard_error": self.limit_standard_error,
            "limit_ci_low": self.limit_ci_low,
            "limit_ci_high": self.limit_ci_high,
            "display_limit_estimate": self.display_limit_estimate,
            "display_limit_ci": self.display_limit_ci,
            "condition_number": self.condition_number,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        level = 100.0 * self.confidence_level
        display_low, display_high = self.display_limit_ci
        lines = [
            "Egger regression test for funnel-plot asymmetry",
            f"Studies: {self.k}",
            (
                f"Asymmetry intercept: {self.intercept:.6g} "
                f"({level:g}% CI {self.intercept_ci_low:.6g} to "
                f"{self.intercept_ci_high:.6g})"
            ),
            f"t({self.df})={self.statistic:.6g}, p={self.pvalue:.6g}",
            (
                "Limit estimate as standard error approaches zero: "
                f"{self.display_limit_estimate:.6g} "
                f"({level:g}% CI {display_low:.6g} to {display_high:.6g})"
            ),
            "Model: weighted regression with multiplicative dispersion",
        ]
        if self.warnings:
            lines.append("Notes:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _validate_confidence_level(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0.0 < float(value) < 1.0
    ):
        raise InvalidStudyDataError("confidence_level must be between 0 and 1.")
    return float(value)


def _fit_classic_egger(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
) -> tuple[float, float, float, float, float, float]:
    """Return intercept, limit, their SEs, t statistic, and condition number."""

    k = len(effect)
    if k < 3:
        raise InsufficientStudiesError(
            "Egger's regression test requires at least three included studies."
        )

    standard_error = np.sqrt(variance)
    # A common positive rescaling of WLS weights leaves both coefficients and
    # their multiplicative-dispersion covariance unchanged. Forming the square-
    # root ratio directly avoids overflowing raw inverse-variance weights.
    weight_root_scale = float(np.min(standard_error))
    root_relative_weight = weight_root_scale / standard_error
    design = np.column_stack((np.ones(k, dtype=np.float64), standard_error))
    weighted_design = root_relative_weight[:, np.newaxis] * design
    weighted_effect = root_relative_weight * effect

    column_scale = np.linalg.norm(weighted_design, axis=0)
    if np.any(~np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        raise InvalidStudyDataError(
            "Egger regression could not construct a finite weighted design."
        )
    standardized_design = weighted_design / column_scale
    try:
        left, singular_values, right_transpose = np.linalg.svd(
            standardized_design,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as error:
        raise InvalidStudyDataError(
            "Egger regression design decomposition did not converge."
        ) from error

    cutoff = (
        float(singular_values[0])
        * max(standardized_design.shape)
        * np.finfo(np.float64).eps
    )
    if len(singular_values) != 2 or singular_values[-1] <= cutoff:
        raise InvalidStudyDataError(
            "Egger's regression test requires standard errors with enough "
            "variation to identify the asymmetry coefficient."
        )

    standardized_coefficients = right_transpose.T @ (
        (left.T @ weighted_effect) / singular_values
    )
    coefficients = standardized_coefficients / column_scale
    fitted = design @ coefficients
    residual = effect - fitted
    weighted_residual = root_relative_weight * residual
    residual_df = k - 2
    residual_dispersion = float(
        np.dot(weighted_residual, weighted_residual) / residual_df
    )
    if not np.isfinite(residual_dispersion) or residual_dispersion <= 0.0:
        raise InvalidStudyDataError(
            "Egger's regression test requires positive finite residual dispersion."
        )

    inverse_information_standardized = (
        right_transpose.T * (1.0 / (singular_values * singular_values))
    ) @ right_transpose
    inverse_column_scale = 1.0 / column_scale
    covariance = residual_dispersion * (
        inverse_column_scale[:, np.newaxis]
        * inverse_information_standardized
        * inverse_column_scale[np.newaxis, :]
    )
    coefficient_standard_errors = np.sqrt(np.diag(covariance))
    if (
        np.any(~np.isfinite(coefficients))
        or np.any(~np.isfinite(coefficient_standard_errors))
        or np.any(coefficient_standard_errors <= 0.0)
    ):
        raise InvalidStudyDataError(
            "Egger regression produced non-finite coefficients or standard errors."
        )

    # In the equivalent y ~ 1 + SE weighted regression, the intercept is the
    # limit estimate and the SE coefficient is Egger's SND-scale intercept.
    limit_estimate = float(coefficients[0])
    intercept = float(coefficients[1])
    limit_standard_error = float(coefficient_standard_errors[0])
    intercept_standard_error = float(coefficient_standard_errors[1])
    statistic = intercept / intercept_standard_error
    condition_number = float(singular_values[0] / singular_values[-1])
    return (
        intercept,
        intercept_standard_error,
        limit_estimate,
        limit_standard_error,
        statistic,
        condition_number,
    )


def egger_test(
    result: MetaAnalysisResult,
    *,
    confidence_level: float | None = None,
) -> EggerTestResult:
    """Run the classical Egger regression test on a fitted meta-analysis."""

    level = _validate_confidence_level(
        result.method.confidence_level if confidence_level is None else confidence_level
    )
    studies = result._study_results_view()
    included = studies["included"].to_numpy(dtype=bool, copy=True)
    effect = studies.loc[included, "effect"].to_numpy(dtype="float64", copy=True)
    variance = studies.loc[included, "variance"].to_numpy(dtype="float64", copy=True)
    (
        intercept,
        intercept_standard_error,
        limit_estimate,
        limit_standard_error,
        statistic,
        condition_number,
    ) = _fit_classic_egger(effect, variance)

    df = len(effect) - 2
    critical_value = float(t.ppf(0.5 + level / 2.0, df=df))
    intercept_margin = critical_value * intercept_standard_error
    limit_margin = critical_value * limit_standard_error
    pvalue = float(2.0 * t.sf(abs(statistic), df=df))

    warnings: list[str] = []
    if len(effect) < 10:
        warnings.append(
            "Funnel-asymmetry tests have low power with fewer than ten studies."
        )
    if result.measure == "OR":
        warnings.append(
            "For odds ratios, inherent association between study effects and their "
            "standard errors can create artifactual Egger-test asymmetry; consider "
            "an outcome-specific method such as Harbord's or Peters' test."
        )
    if result.measure == "SMD":
        warnings.append(
            "For standardized mean differences, inherent association between study "
            "effects and their standard errors can create artifactual Egger-test "
            "asymmetry."
        )
    warnings.append(
        "This test diagnoses a relationship between effect estimates and standard "
        "errors; it does not by itself establish or exclude publication bias."
    )

    return EggerTestResult(
        intercept=intercept,
        intercept_standard_error=intercept_standard_error,
        intercept_ci_low=intercept - intercept_margin,
        intercept_ci_high=intercept + intercept_margin,
        statistic=statistic,
        statistic_name="t",
        distribution="t",
        df=df,
        pvalue=pvalue,
        limit_estimate=limit_estimate,
        limit_standard_error=limit_standard_error,
        limit_ci_low=limit_estimate - limit_margin,
        limit_ci_high=limit_estimate + limit_margin,
        confidence_level=level,
        k=len(effect),
        measure=result.measure,
        effect_scale=result.effect_scale,
        display_scale=result.display_scale,
        method="egger",
        model="weighted_regression_multiplicative_dispersion",
        predictor="standard_error",
        condition_number=condition_number,
        warnings=tuple(warnings),
    )
