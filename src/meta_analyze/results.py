"""Immutable, inspectable result objects."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pandas as pd

from .config import MethodConfig, SubgroupMethodConfig

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


@dataclass(frozen=True, slots=True)
class HeterogeneityResult:
    """Classical heterogeneity statistics."""

    q: float
    df: int
    pvalue: float
    i2: float
    h2: float


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    """Convergence details for the fitted model."""

    converged: bool
    iterations: int
    tau2_at_boundary: bool | None


@dataclass(frozen=True, slots=True)
class MetaAnalysisSummary:
    """A printable summary with a machine-readable representation."""

    _result: MetaAnalysisResult = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = self._result
        return {
            "model": result.model,
            "pooling_method": result.method.pooling_method,
            "measure": result.measure,
            "effect_scale": result.effect_scale,
            "display_scale": result.display_scale,
            "studies": result.k,
            "estimate": result.estimate,
            "display_estimate": result.display_estimate,
            "standard_error": result.standard_error,
            "confidence_level": result.method.confidence_level,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "display_ci": result.display_ci,
            "prediction_interval": result.prediction_interval,
            "display_prediction_interval": result.display_prediction_interval,
            "tau2": result.tau2,
            "tau2_method": result.method.tau2_method,
            "q": result.q,
            "q_df": result.q_df,
            "q_pvalue": result.q_pvalue,
            "i2": result.i2,
            "h2": result.h2,
            "warnings": result.warnings,
            "method_options": dict(result.method.options),
        }

    def __str__(self) -> str:
        result = self._result
        level = 100.0 * result.method.confidence_level
        display_low, display_high = result.display_ci
        title = f"Meta-analysis ({result.model}-effect, {result.measure})"
        lines = [
            title,
            f"Studies: {result.k}",
            (
                f"Estimate: {result.display_estimate:.6g} "
                f"({level:g}% CI {display_low:.6g} to {display_high:.6g})"
            ),
        ]
        if result.model == "random":
            lines.append(f"tau^2: {result.tau2:.6g} ({result.method.tau2_method})")
            if result.prediction_interval is not None:
                display_prediction = result.display_prediction_interval
                assert display_prediction is not None
                low, high = display_prediction
                lines.append(f"Prediction interval: {low:.6g} to {high:.6g}")

        if result.q_df > 0:
            lines.extend(
                [
                    (f"Q({result.q_df}): {result.q:.6g}, p={result.q_pvalue:.6g}"),
                    f"I^2: {100.0 * result.i2:.2f}%; H^2: {result.h2:.6g}",
                ]
            )
        else:
            lines.append("Heterogeneity: not estimable with one study")

        lines.append(f"Confidence interval method: {result.method.ci_method}")
        if result.warnings:
            lines.append("Notes:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SubgroupMetaAnalysisSummary:
    """A printable and machine-readable subgroup analysis summary."""

    _result: SubgroupMetaAnalysisResult = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return nested subgroup, overall, and interaction-test results."""

        result = self._result
        return {
            "groups": {
                label: group.summary().to_dict()
                for label, group in result.groups.items()
            },
            "overall": result.overall.summary().to_dict(),
            "q_between": result.q_between,
            "q_between_df": result.q_between_df,
            "q_between_pvalue": result.q_between_pvalue,
            "i2_between": result.i2_between,
            "method": {
                "model": result.method.model,
                "tau2_strategy": result.method.tau2_strategy,
                "test_method": result.method.test_method,
                "subgroup_missing": result.method.subgroup_missing,
            },
            "warnings": result.warnings,
        }

    def __str__(self) -> str:
        result = self._result
        level = 100.0 * result.overall.method.confidence_level
        lines = [
            f"Subgroup meta-analysis ({result.overall.model}-effect, "
            f"{result.overall.measure})",
            f"Groups: {len(result.groups)}; studies: {result.overall.k}",
        ]
        for label, group in result.groups.items():
            low, high = group.display_ci
            lines.append(
                f"- {label}: {group.display_estimate:.6g} "
                f"({level:g}% CI {low:.6g} to {high:.6g}; k={group.k})"
            )
        lines.append(
            "Test for subgroup differences: "
            f"Q({result.q_between_df})={result.q_between:.6g}, "
            f"p={result.q_between_pvalue:.6g}, "
            f"I^2={100.0 * result.i2_between:.2f}%"
        )
        lines.append(
            f"Subgroup tau^2 strategy: {result.method.tau2_strategy}; "
            f"test method: {result.method.test_method}"
        )
        if result.warnings:
            lines.append("Notes:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MetaAnalysisResult:
    """Results of a generic inverse-variance meta-analysis."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    prediction_interval: tuple[float, float] | None
    tau2: float
    heterogeneity: HeterogeneityResult
    k: int
    model: str
    measure: str
    effect_scale: str
    display_scale: str
    method: MethodConfig
    diagnostics: FitDiagnostics
    warnings: tuple[str, ...]
    _study_results: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_study_results", self._study_results.copy(deep=True))

    @property
    def ci(self) -> tuple[float, float]:
        return self.ci_low, self.ci_high

    def _to_display_scale(self, value: float) -> float:
        if self.display_scale == "identity":
            return value
        if self.display_scale == "exp":
            return math.exp(value)
        raise ValueError(f"Unknown display scale {self.display_scale!r}.")

    @property
    def display_estimate(self) -> float:
        return self._to_display_scale(self.estimate)

    @property
    def display_ci(self) -> tuple[float, float]:
        return (
            self._to_display_scale(self.ci_low),
            self._to_display_scale(self.ci_high),
        )

    @property
    def display_prediction_interval(self) -> tuple[float, float] | None:
        if self.prediction_interval is None:
            return None
        low, high = self.prediction_interval
        return self._to_display_scale(low), self._to_display_scale(high)

    @property
    def q(self) -> float:
        return self.heterogeneity.q

    @property
    def q_df(self) -> int:
        return self.heterogeneity.df

    @property
    def q_pvalue(self) -> float:
        return self.heterogeneity.pvalue

    @property
    def i2(self) -> float:
        """I-squared as a proportion from 0 to 1."""

        return self.heterogeneity.i2

    @property
    def h2(self) -> float:
        return self.heterogeneity.h2

    @property
    def study_results(self) -> pd.DataFrame:
        """Return a defensive copy of row-level inputs and fitted weights."""

        return self._study_results.copy(deep=True)

    @property
    def excluded_studies(self) -> pd.DataFrame:
        studies = self._study_results
        return studies.loc[~studies["included"]].copy(deep=True)

    def summary(self) -> MetaAnalysisSummary:
        return MetaAnalysisSummary(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the same row-level table as :attr:`study_results`."""

        return self.study_results

    def forest(
        self,
        *,
        ax: Axes | None = None,
        effect_label: str | None = None,
        pooled_label: str | None = None,
        show_prediction_interval: bool = True,
        show_weights: bool = True,
        null_value: float | None = None,
        log_scale: bool | None = None,
    ) -> Axes:
        """Draw a Matplotlib forest plot without calling ``show()``.

        Matplotlib is an optional dependency. Install ``PyMetaAnalysis[plot]``
        before calling this method.
        """

        from .plotting import forest_plot

        return forest_plot(
            self,
            ax=ax,
            effect_label=effect_label,
            pooled_label=pooled_label,
            show_prediction_interval=show_prediction_interval,
            show_weights=show_weights,
            null_value=null_value,
            log_scale=log_scale,
        )

    def funnel(
        self,
        *,
        ax: Axes | None = None,
        effect_label: str | None = None,
        confidence_level: float | None = None,
        show_pseudo_confidence_interval: bool = True,
        warn_on_few_studies: bool = True,
        log_scale: bool | None = None,
    ) -> Axes:
        """Draw a standard-error funnel plot without calling ``show()``.

        Matplotlib is an optional dependency. Install ``PyMetaAnalysis[plot]``
        before calling this method. Funnel asymmetry is a diagnostic for
        possible small-study effects and does not by itself establish
        publication bias.
        """

        from .plotting import funnel_plot

        return funnel_plot(
            self,
            ax=ax,
            effect_label=effect_label,
            confidence_level=confidence_level,
            show_pseudo_confidence_interval=show_pseudo_confidence_interval,
            warn_on_few_studies=warn_on_few_studies,
            log_scale=log_scale,
        )


@dataclass(frozen=True, slots=True)
class SubgroupMetaAnalysisResult:
    """Results for independent study subgroups and their formal comparison."""

    groups: Mapping[Hashable, MetaAnalysisResult]
    overall: MetaAnalysisResult
    q_between: float
    q_between_df: int
    q_between_pvalue: float
    i2_between: float
    method: SubgroupMethodConfig
    warnings: tuple[str, ...]
    _study_results: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Defensively freeze the group mapping and combined study table."""

        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))
        object.__setattr__(self, "_study_results", self._study_results.copy(deep=True))

    @property
    def study_results(self) -> pd.DataFrame:
        """Return all rows with their subgroup labels and overall fit metadata."""

        return self._study_results.copy(deep=True)

    @property
    def excluded_studies(self) -> pd.DataFrame:
        """Return rows excluded from the overall and subgroup fits."""

        studies = self._study_results
        return studies.loc[~studies["included"]].copy(deep=True)

    def summary(self) -> SubgroupMetaAnalysisSummary:
        """Return a printable and machine-readable subgroup summary."""

        return SubgroupMetaAnalysisSummary(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the combined row-level subgroup table."""

        return self.study_results

    def forest(
        self,
        *,
        ax: Axes | None = None,
        effect_label: str | None = None,
        show_prediction_interval: bool = True,
        show_weights: bool = True,
        null_value: float | None = None,
        log_scale: bool | None = None,
    ) -> Axes:
        """Draw a subgroup forest plot without calling ``show()``."""

        from .plotting import subgroup_forest_plot

        return subgroup_forest_plot(
            self,
            ax=ax,
            effect_label=effect_label,
            show_prediction_interval=show_prediction_interval,
            show_weights=show_weights,
            null_value=null_value,
            log_scale=log_scale,
        )
