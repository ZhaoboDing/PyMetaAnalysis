"""Duval and Tweedie trim-and-fill sensitivity analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from .exceptions import (
    ConvergenceError,
    InsufficientStudiesError,
    UnsupportedMethodError,
)
from .provenance import TransformationRecord

if TYPE_CHECKING:
    from .results import MetaAnalysisResult

TrimFillSide = Literal["left", "right"]
TrimFillEstimator = Literal["L0", "R0"]


@dataclass(frozen=True, slots=True)
class TrimAndFillResult:
    """Result of a Duval and Tweedie trim-and-fill sensitivity analysis."""

    adjusted_result: MetaAnalysisResult
    original_result: MetaAnalysisResult
    side: TrimFillSide
    estimator: TrimFillEstimator
    k0: int
    k0_standard_error: float
    k0_pvalue: float | None
    iterations: int
    converged: bool
    max_iterations: int
    original_estimate: float
    adjusted_estimate: float
    original_tau2: float
    adjusted_tau2: float
    warnings: tuple[str, ...]
    _augmented_studies: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_augmented_studies", self._augmented_studies.copy(deep=True)
        )

    @property
    def augmented_studies(self) -> pd.DataFrame:
        """Return observed and imputed studies as a defensive copy."""
        return self._augmented_studies.copy(deep=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, machine-readable summary."""
        return {
            "method": "trim_and_fill",
            "side": self.side,
            "estimator": self.estimator,
            "k0": self.k0,
            "k0_standard_error": self.k0_standard_error,
            "k0_pvalue": self.k0_pvalue,
            "iterations": self.iterations,
            "converged": self.converged,
            "max_iterations": self.max_iterations,
            "original_estimate": self.original_estimate,
            "adjusted_estimate": self.adjusted_estimate,
            "original_tau2": self.original_tau2,
            "adjusted_tau2": self.adjusted_tau2,
            "warnings": self.warnings,
            "augmented_studies": self.augmented_studies.to_dict(orient="records"),
        }

    def __str__(self) -> str:
        lines = [
            "Duval and Tweedie trim-and-fill analysis",
            f"Estimated missing studies: {self.k0} ({self.side}; {self.estimator})",
            f"Original estimate: {self.original_estimate:.6g}",
            f"Adjusted estimate: {self.adjusted_estimate:.6g}",
            f"Iterations: {self.iterations}/{self.max_iterations}",
        ]
        if self.warnings:
            lines.extend(("Notes:", *(f"- {warning}" for warning in self.warnings)))
        return "\n".join(lines)


def _fit(
    result: MetaAnalysisResult,
    effect: np.ndarray,
    variance: np.ndarray,
    study: np.ndarray,
) -> MetaAnalysisResult:
    from .api import meta_analysis

    return meta_analysis(
        effect=effect,
        variance=variance,
        study=study,
        model=result.model,
        tau2_method=result.method.tau2_method,
        ci_method=result.method.ci_method,
        confidence_level=result.method.confidence_level,
        missing="raise",
        atol=result.method.atol,
        max_iter=result.method.max_iter,
    )


def _automatic_side(
    result: MetaAnalysisResult, effect: np.ndarray, variance: np.ndarray
) -> TrimFillSide:
    from .regression_api import meta_regression

    if len(effect) < 3:
        raise InsufficientStudiesError(
            "Automatic side selection requires at least 3 studies; "
            "specify side explicitly."
        )
    regression = meta_regression(
        effect=effect,
        variance=variance,
        moderators={"sei": np.sqrt(variance)},
        model="common" if result.model == "common" else "mixed",
        tau2_method=result.method.tau2_method,
        inference_method="normal",
        atol=result.method.atol,
        max_iter=result.method.max_iter,
    )
    return "right" if float(regression.coefficients.iloc[1]["estimate"]) < 0 else "left"


def _rank_first(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def trim_and_fill(
    result: MetaAnalysisResult,
    *,
    side: TrimFillSide | None = None,
    estimator: TrimFillEstimator = "L0",
    max_iterations: int = 100,
) -> TrimAndFillResult:
    """Apply the Duval and Tweedie L0 or R0 trim-and-fill procedure."""
    if result.method.pooling_method != "inverse_variance":
        raise UnsupportedMethodError(
            "trim_and_fill() requires inverse-variance pooling."
        )
    if side not in (None, "left", "right"):
        raise UnsupportedMethodError("side must be 'left', 'right', or None.")
    if not isinstance(estimator, str):
        raise TypeError("estimator must be a string.")
    normalized_estimator = estimator.upper()
    if normalized_estimator not in ("L0", "R0"):
        raise UnsupportedMethodError("estimator must be 'L0' or 'R0'.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral):
        raise TypeError("max_iterations must be an integer.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    studies = result._study_results_view()
    included = studies["included"].to_numpy(dtype=bool, copy=True)
    effect0 = studies.loc[included, "effect"].to_numpy(dtype=float, copy=True)
    variance0 = studies.loc[included, "variance"].to_numpy(dtype=float, copy=True)
    labels0 = studies.loc[included, "study"].to_numpy(dtype=object, copy=True)
    k = len(effect0)
    if k < 2:
        raise InsufficientStudiesError("trim_and_fill() requires at least 2 studies.")
    resolved_side = (
        _automatic_side(result, effect0, variance0) if side is None else side
    )
    working = -effect0 if resolved_side == "right" else effect0
    order = np.argsort(working, kind="stable")
    yi, vi, labels = working[order], variance0[order], labels0[order]

    previous, k0, k0_se, iterations = -1, 0, math.sqrt(2.0), 0
    center, centered = result.estimate, yi - result.estimate
    while k0 != previous:
        previous, iterations = k0, iterations + 1
        if iterations > max_iterations:
            raise ConvergenceError(
                f"Trim-and-fill did not converge within {max_iterations} iterations."
            )
        if k - k0 < 1:
            raise ConvergenceError("Trim-and-fill removed every study.")
        center = _fit(result, yi[: k - k0], vi[: k - k0], labels[: k - k0]).estimate
        centered = yi - center
        signed = np.sign(centered) * _rank_first(np.abs(centered))
        if normalized_estimator == "R0":
            negative = signed[signed < 0]
            raw = (k - (float(np.max(-negative)) if len(negative) else 0.0)) - 1.0
            k0_se = math.sqrt(2.0 * max(0.0, raw) + 2.0)
        else:
            sr = float(np.sum(signed[signed > 0]))
            raw = (4.0 * sr - k * (k + 1.0)) / (2.0 * k - 1.0)
            var_sr = (
                k * (k + 1.0) * (2.0 * k + 1.0)
                + 10.0 * raw**3
                + 27.0 * raw**2
                + 17.0 * raw
                - 18.0 * k * raw**2
                - 18.0 * k * raw
                + 6.0 * k**2 * raw
            ) / 24.0
            k0_se = 4.0 * math.sqrt(max(0.0, var_sr)) / (2.0 * k - 1.0)
        k0 = max(0, int(np.rint(raw)))

    if k0:
        tail = slice(k - k0, k)
        mirror_effect = (
            centered[tail] - center
            if resolved_side == "right"
            else -centered[tail] + center
        )
        mirror_variance, mirror_labels = vi[tail].copy(), labels[tail].copy()
        filled_labels = np.asarray(
            [f"Filled {i}" for i in range(1, k0 + 1)], dtype=object
        )
        adjusted = _fit(
            result,
            np.concatenate((effect0, mirror_effect)),
            np.concatenate((variance0, mirror_variance)),
            np.concatenate((labels0, filled_labels)),
        )
    else:
        mirror_labels = np.empty(0, dtype=object)
        adjusted = result

    table = adjusted._study_results_view().copy(deep=True)
    table["imputed"] = np.r_[np.zeros(k, dtype=bool), np.ones(k0, dtype=bool)]
    table["mirror_source"] = np.r_[np.full(k, None, dtype=object), mirror_labels]
    synthetic_rows = tuple(range(k, k + k0))
    provenance = replace(
        result.provenance,
        data_source="derived_trim_and_fill",
        row_count=k + k0,
        included_rows=tuple(range(k + k0)),
        excluded_rows=(),
        transformations=(
            *result.provenance.transformations,
            TransformationRecord(
                name="trim_and_fill_imputation",
                parameters=(
                    ("side", resolved_side),
                    ("estimator", normalized_estimator),
                    ("k0", k0),
                ),
                affected_rows=synthetic_rows,
            ),
        ),
    )
    adjusted = replace(
        adjusted,
        measure=result.measure,
        effect_scale=result.effect_scale,
        display_scale=result.display_scale,
        method=result.method,
        provenance=provenance,
        warnings=(
            *adjusted.warnings,
            "Trim-and-fill estimates are exploratory and assume funnel symmetry.",
        ),
        _study_results=table,
        _source_data=None,
    )
    return TrimAndFillResult(
        adjusted_result=adjusted,
        original_result=result,
        side=resolved_side,
        estimator=cast("TrimFillEstimator", normalized_estimator),
        k0=k0,
        k0_standard_error=max(0.0, k0_se),
        k0_pvalue=2.0 ** (-(k0 + 1)) if normalized_estimator == "R0" else None,
        iterations=iterations,
        converged=True,
        max_iterations=max_iterations,
        original_estimate=result.estimate,
        adjusted_estimate=adjusted.estimate,
        original_tau2=result.tau2,
        adjusted_tau2=adjusted.tau2,
        warnings=(
            "Trim-and-fill is a sensitivity analysis, not proof of publication bias.",
        ),
        _augmented_studies=table,
    )
