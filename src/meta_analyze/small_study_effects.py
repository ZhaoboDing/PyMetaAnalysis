"""Regression diagnostics for funnel-plot asymmetry and small-study effects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t

from .exceptions import (
    InsufficientStudiesError,
    InvalidStudyDataError,
    UnsupportedMethodError,
)

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


@dataclass(frozen=True, slots=True)
class PetersTestResult:
    """Peters regression test for small-study effects in odds ratios."""

    slope: float
    slope_standard_error: float
    slope_ci_low: float
    slope_ci_high: float
    statistic: float
    statistic_name: str
    distribution: str
    df: int
    pvalue: float
    limit_estimate: float
    limit_standard_error: float
    limit_ci_low: float
    limit_ci_high: float
    residual_dispersion: float
    confidence_level: float
    k: int
    measure: str
    effect_scale: str
    display_scale: str
    method: str
    model: str
    predictor: str
    weight_method: str
    continuity_correction: float
    correction_scope: str
    corrected_studies: int
    condition_number: float
    warnings: tuple[str, ...]

    @property
    def slope_ci(self) -> tuple[float, float]:
        """Return the confidence interval for the asymmetry slope."""

        return self.slope_ci_low, self.slope_ci_high

    @property
    def limit_ci(self) -> tuple[float, float]:
        """Return the log-odds-ratio interval for the limit estimate."""

        return self.limit_ci_low, self.limit_ci_high

    @property
    def display_limit_estimate(self) -> float:
        """Return the limit estimate on the odds-ratio scale."""

        return math.exp(self.limit_estimate)

    @property
    def display_limit_ci(self) -> tuple[float, float]:
        """Return the limit-estimate interval on the odds-ratio scale."""

        return math.exp(self.limit_ci_low), math.exp(self.limit_ci_high)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, machine-readable representation."""

        return {
            "method": self.method,
            "model": self.model,
            "predictor": self.predictor,
            "weight_method": self.weight_method,
            "studies": self.k,
            "measure": self.measure,
            "effect_scale": self.effect_scale,
            "display_scale": self.display_scale,
            "confidence_level": self.confidence_level,
            "slope": self.slope,
            "slope_standard_error": self.slope_standard_error,
            "slope_ci_low": self.slope_ci_low,
            "slope_ci_high": self.slope_ci_high,
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
            "residual_dispersion": self.residual_dispersion,
            "continuity_correction": self.continuity_correction,
            "correction_scope": self.correction_scope,
            "corrected_studies": self.corrected_studies,
            "condition_number": self.condition_number,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        level = 100.0 * self.confidence_level
        display_low, display_high = self.display_limit_ci
        lines = [
            "Peters regression test for funnel-plot asymmetry",
            f"Studies: {self.k}",
            (
                f"Asymmetry slope: {self.slope:.6g} "
                f"({level:g}% CI {self.slope_ci_low:.6g} to "
                f"{self.slope_ci_high:.6g})"
            ),
            f"t({self.df})={self.statistic:.6g}, p={self.pvalue:.6g}",
            (
                "Limit odds ratio as total sample size approaches infinity: "
                f"{self.display_limit_estimate:.6g} "
                f"({level:g}% CI {display_low:.6g} to {display_high:.6g})"
            ),
            "Model: weighted regression with multiplicative dispersion",
        ]
        if self.warnings:
            lines.append("Notes:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _WeightedRegressionFit:
    intercept: float
    intercept_standard_error: float
    slope: float
    slope_standard_error: float
    slope_statistic: float
    relative_residual_dispersion: float
    condition_number: float


def _validate_confidence_level(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0.0 < float(value) < 1.0
    ):
        raise InvalidStudyDataError("confidence_level must be between 0 and 1.")
    return float(value)


def _fit_weighted_regression(
    effect: NDArray[np.float64],
    predictor: NDArray[np.float64],
    root_relative_weight: NDArray[np.float64],
    *,
    diagnostic_name: str,
    predictor_label: str,
) -> _WeightedRegressionFit:
    """Fit a two-coefficient WLS model with multiplicative dispersion."""

    k = len(effect)
    if k < 3:
        raise InsufficientStudiesError(
            f"{diagnostic_name} requires at least three included studies."
        )
    if (
        len(predictor) != k
        or len(root_relative_weight) != k
        or np.any(~np.isfinite(effect))
        or np.any(~np.isfinite(predictor))
        or np.any(~np.isfinite(root_relative_weight))
        or np.any(root_relative_weight <= 0.0)
    ):
        raise InvalidStudyDataError(
            f"{diagnostic_name} requires finite effects, predictors, and "
            "positive weights."
        )

    design = np.column_stack((np.ones(k, dtype=np.float64), predictor))
    weighted_design = root_relative_weight[:, np.newaxis] * design
    weighted_effect = root_relative_weight * effect

    column_scale = np.linalg.norm(weighted_design, axis=0)
    if np.any(~np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        raise InvalidStudyDataError(
            f"{diagnostic_name} could not construct a finite weighted design."
        )
    standardized_design = weighted_design / column_scale
    try:
        left, singular_values, right_transpose = np.linalg.svd(
            standardized_design,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as error:
        raise InvalidStudyDataError(
            f"{diagnostic_name} design decomposition did not converge."
        ) from error

    cutoff = (
        float(singular_values[0])
        * max(standardized_design.shape)
        * np.finfo(np.float64).eps
    )
    if len(singular_values) != 2 or singular_values[-1] <= cutoff:
        raise InvalidStudyDataError(
            f"{diagnostic_name} requires {predictor_label} with enough "
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
    relative_residual_dispersion = float(
        np.dot(weighted_residual, weighted_residual) / residual_df
    )
    if (
        not np.isfinite(relative_residual_dispersion)
        or relative_residual_dispersion <= 0.0
    ):
        raise InvalidStudyDataError(
            f"{diagnostic_name} requires positive finite residual dispersion."
        )

    inverse_information_standardized = (
        right_transpose.T * (1.0 / (singular_values * singular_values))
    ) @ right_transpose
    inverse_column_scale = 1.0 / column_scale
    covariance = relative_residual_dispersion * (
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
            f"{diagnostic_name} produced non-finite coefficients or standard errors."
        )

    intercept = float(coefficients[0])
    slope = float(coefficients[1])
    intercept_standard_error = float(coefficient_standard_errors[0])
    slope_standard_error = float(coefficient_standard_errors[1])
    return _WeightedRegressionFit(
        intercept=intercept,
        intercept_standard_error=intercept_standard_error,
        slope=slope,
        slope_standard_error=slope_standard_error,
        slope_statistic=slope / slope_standard_error,
        relative_residual_dispersion=relative_residual_dispersion,
        condition_number=float(singular_values[0] / singular_values[-1]),
    )


def _fit_classic_egger(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
) -> tuple[float, float, float, float, float, float]:
    """Return intercept, limit, their SEs, t statistic, and condition number."""

    standard_error = np.sqrt(variance)
    # A common positive rescaling of WLS weights leaves both coefficients and
    # their multiplicative-dispersion covariance unchanged. Forming the square-
    # root ratio directly avoids overflowing raw inverse-variance weights.
    weight_root_scale = float(np.min(standard_error))
    root_relative_weight = weight_root_scale / standard_error
    fit = _fit_weighted_regression(
        effect,
        standard_error,
        root_relative_weight,
        diagnostic_name="Egger's regression test",
        predictor_label="standard errors",
    )

    # In the equivalent y ~ 1 + SE weighted regression, the intercept is the
    # limit estimate and the SE coefficient is Egger's SND-scale intercept.
    return (
        fit.slope,
        fit.slope_standard_error,
        fit.intercept,
        fit.intercept_standard_error,
        fit.slope_statistic,
        fit.condition_number,
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


_PETERS_REQUIRED_COLUMNS = frozenset(
    {
        "event_treat",
        "n_treat",
        "event_control",
        "n_control",
        "included",
        "continuity_corrected",
    }
)


def peters_test(
    result: MetaAnalysisResult,
    *,
    confidence_level: float | None = None,
) -> PetersTestResult:
    """Run Peters' regression test for small-study effects in odds ratios."""

    if result.measure != "OR":
        raise UnsupportedMethodError(
            "Peters' regression test is implemented only for measure='OR'."
        )
    studies = result._study_results_view()
    missing_columns = sorted(_PETERS_REQUIRED_COLUMNS.difference(studies.columns))
    if missing_columns:
        raise UnsupportedMethodError(
            "Peters' regression test requires a result produced by meta_binary() "
            "with retained two-group counts; missing study fields: "
            + ", ".join(missing_columns)
            + "."
        )

    options = dict(result.method.options)
    correction_value = options.get("continuity_correction")
    correction_scope = options.get("correction_scope")
    if (
        isinstance(correction_value, bool)
        or not isinstance(correction_value, Real)
        or not np.isfinite(correction_value)
        or float(correction_value) < 0.0
        or not isinstance(correction_scope, str)
        or correction_scope
        not in {"only_zero_studies", "all_studies", "if_any_zero", "none"}
    ):
        raise InvalidStudyDataError(
            "Peters' regression test requires a valid recorded study-level "
            "continuity-correction contract."
        )
    correction = float(correction_value)
    level = _validate_confidence_level(
        result.method.confidence_level if confidence_level is None else confidence_level
    )

    included = studies["included"].to_numpy(dtype=bool, copy=True)
    included_studies = studies.loc[included]
    if len(included_studies) < 3:
        raise InsufficientStudiesError(
            "Peters' regression test requires at least three included studies."
        )
    event_treat = included_studies["event_treat"].to_numpy(dtype=np.float64, copy=True)
    n_treat = included_studies["n_treat"].to_numpy(dtype=np.float64, copy=True)
    event_control = included_studies["event_control"].to_numpy(
        dtype=np.float64, copy=True
    )
    n_control = included_studies["n_control"].to_numpy(dtype=np.float64, copy=True)
    corrected = included_studies["continuity_corrected"].to_numpy(dtype=bool, copy=True)

    a = event_treat
    b = n_treat - event_treat
    c = event_control
    d = n_control - event_control
    total_sample_size = n_treat + n_control
    total_events = a + c
    total_nonevents = b + d
    if (
        np.any(~np.isfinite(total_sample_size))
        or np.any(total_sample_size <= 0.0)
        or np.any(~np.isfinite(total_events))
        or np.any(total_events <= 0.0)
        or np.any(~np.isfinite(total_nonevents))
        or np.any(total_nonevents <= 0.0)
    ):
        raise InvalidStudyDataError(
            "Peters' regression test requires positive finite total sample sizes, "
            "total events, and total non-events in every included study."
        )

    if np.any(corrected):
        for cell in (a, b, c, d):
            cell[corrected] += correction
    if any(np.any(~np.isfinite(cell) | (cell <= 0.0)) for cell in (a, b, c, d)):
        raise InvalidStudyDataError(
            "Peters' regression test requires positive four-cell counts after the "
            "recorded study-level continuity correction; refit meta_binary() with "
            "a positive continuity_correction and a correction_scope that selects "
            "the zero-cell studies."
        )

    log_odds_ratio = np.log(a) + np.log(d) - np.log(b) - np.log(c)
    predictor = 1.0 / total_sample_size
    # S*F/N is evaluated with the smaller margin first so the intermediate
    # product cannot exceed the original finite counts.
    smaller_margin = np.minimum(total_events, total_nonevents)
    larger_margin = np.maximum(total_events, total_nonevents)
    weight = smaller_margin * (larger_margin / total_sample_size)
    if np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
        raise InvalidStudyDataError(
            "Peters' regression test produced non-positive or non-finite S*F/N weights."
        )
    maximum_weight = float(np.max(weight))
    root_relative_weight = np.sqrt(weight / maximum_weight)
    fit = _fit_weighted_regression(
        log_odds_ratio,
        predictor,
        root_relative_weight,
        diagnostic_name="Peters' regression test",
        predictor_label="total sample sizes",
    )
    residual_dispersion = fit.relative_residual_dispersion * maximum_weight
    if not np.isfinite(residual_dispersion) or residual_dispersion <= 0.0:
        raise InvalidStudyDataError(
            "Peters' regression test produced non-finite residual dispersion."
        )

    df = len(log_odds_ratio) - 2
    critical_value = float(t.ppf(0.5 + level / 2.0, df=df))
    slope_margin = critical_value * fit.slope_standard_error
    limit_margin = critical_value * fit.intercept_standard_error
    pvalue = float(2.0 * t.sf(abs(fit.slope_statistic), df=df))

    warnings: list[str] = []
    if len(log_odds_ratio) < 10:
        warnings.append(
            "Funnel-asymmetry tests have low power with fewer than ten studies."
        )
    if result.method.pooling_method == "peto":
        warnings.append(
            "The source analysis uses Peto pooling; this diagnostic reconstructs "
            "conventional continuity-corrected study log odds ratios from the "
            "retained four-cell counts."
        )
    warnings.extend(
        [
            "Peters' test is designed for comparative two-group binary outcomes "
            "and should not be used for diagnostic-accuracy studies.",
            "This test diagnoses a relationship between log odds ratios and "
            "inverse total sample size; it does not by itself establish or "
            "exclude publication bias.",
        ]
    )

    return PetersTestResult(
        slope=fit.slope,
        slope_standard_error=fit.slope_standard_error,
        slope_ci_low=fit.slope - slope_margin,
        slope_ci_high=fit.slope + slope_margin,
        statistic=fit.slope_statistic,
        statistic_name="t",
        distribution="t",
        df=df,
        pvalue=pvalue,
        limit_estimate=fit.intercept,
        limit_standard_error=fit.intercept_standard_error,
        limit_ci_low=fit.intercept - limit_margin,
        limit_ci_high=fit.intercept + limit_margin,
        residual_dispersion=float(residual_dispersion),
        confidence_level=level,
        k=len(log_odds_ratio),
        measure="OR",
        effect_scale="log",
        display_scale="exp",
        method="peters",
        model="weighted_regression_multiplicative_dispersion",
        predictor="inverse_total_sample_size",
        weight_method="S_times_F_over_N",
        continuity_correction=correction,
        correction_scope=correction_scope,
        corrected_studies=int(np.count_nonzero(corrected)),
        condition_number=fit.condition_number,
        warnings=tuple(warnings),
    )
