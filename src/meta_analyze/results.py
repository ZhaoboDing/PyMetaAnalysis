"""Immutable, inspectable result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import MethodConfig


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
            "measure": result.measure,
            "studies": result.k,
            "estimate": result.estimate,
            "standard_error": result.standard_error,
            "confidence_level": result.method.confidence_level,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "prediction_interval": result.prediction_interval,
            "tau2": result.tau2,
            "tau2_method": result.method.tau2_method,
            "q": result.q,
            "q_df": result.q_df,
            "q_pvalue": result.q_pvalue,
            "i2": result.i2,
            "h2": result.h2,
            "warnings": result.warnings,
        }

    def __str__(self) -> str:
        result = self._result
        level = 100.0 * result.method.confidence_level
        title = f"Meta-analysis ({result.model}-effect, {result.measure})"
        lines = [
            title,
            f"Studies: {result.k}",
            (
                f"Estimate: {result.estimate:.6g} "
                f"({level:g}% CI {result.ci_low:.6g} to {result.ci_high:.6g})"
            ),
        ]
        if result.model == "random":
            lines.append(f"tau^2: {result.tau2:.6g} ({result.method.tau2_method})")
            if result.prediction_interval is not None:
                low, high = result.prediction_interval
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
    method: MethodConfig
    diagnostics: FitDiagnostics
    warnings: tuple[str, ...]
    _study_results: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_study_results", self._study_results.copy(deep=True))

    @property
    def ci(self) -> tuple[float, float]:
        return self.ci_low, self.ci_high

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
