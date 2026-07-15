"""High-level API for two-group continuous outcome meta-analysis."""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

from .api import _normalize_ci_method, _normalize_model, _validate_analysis_controls
from .config import MethodConfig, MethodOptionValue
from .data import ColumnOrArray, MissingPolicy
from .effect_sizes.continuous import (
    calculate_continuous_effects,
    normalize_continuous_studies,
)
from .estimators import fit_inverse_variance
from .heterogeneity import classical_heterogeneity
from .results import (
    FitDiagnostics,
    HeterogeneityResult,
    MetaAnalysisResult,
    SubgroupMetaAnalysisResult,
)
from .subgroups import fit_subgroup_analysis


def _fit_meta_continuous_single(
    data: pd.DataFrame | None = None,
    *,
    mean_treat: ColumnOrArray,
    sd_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    mean_control: ColumnOrArray,
    sd_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None = None,
    measure: str = "MD",
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    smd_variance: str = "LS",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult:
    """Pool mean differences or Hedges' g from two independent groups.

    ``measure="MD"`` uses the unpooled sampling variance
    ``sd_treat**2 / n_treat + sd_control**2 / n_control``. ``measure="SMD"``
    uses a pooled SD, the exact Hedges correction, and the ``LS`` sampling
    variance convention used by ``metafor::escalc(measure="SMD")``.
    """

    confidence_level, atol, max_iter = _validate_analysis_controls(
        confidence_level=confidence_level,
        atol=atol,
        max_iter=max_iter,
    )
    normalized_model = _normalize_model(model)
    normalized_ci = _normalize_ci_method(ci_method)
    normalized_tau2 = tau2_method.upper().replace("-", "_")

    studies = normalize_continuous_studies(
        data=data,
        mean_treat=mean_treat,
        sd_treat=sd_treat,
        n_treat=n_treat,
        mean_control=mean_control,
        sd_control=sd_control,
        n_control=n_control,
        study=study,
        missing=missing,
    )
    effects = calculate_continuous_effects(
        studies,
        measure=measure,
        smd_variance=smd_variance,
    )
    included = studies.included
    included_effect = effects.included_effect
    included_variance = effects.included_variance

    fit = fit_inverse_variance(
        included_effect,
        included_variance,
        model=normalized_model,
        tau2_method=normalized_tau2,
        ci_method=normalized_ci,
        confidence_level=confidence_level,
        atol=atol,
        max_iter=max_iter,
    )
    q, q_df, q_pvalue, i2, h2 = classical_heterogeneity(
        included_effect, included_variance
    )
    heterogeneity = HeterogeneityResult(q, q_df, q_pvalue, i2, h2)

    row_count = len(included)
    raw_weights = np.full(row_count, np.nan, dtype=np.float64)
    normalized_weights = np.full(row_count, np.nan, dtype=np.float64)
    raw_weights[included] = fit.weights
    normalized_weights[included] = fit.normalized_weights
    study_results = pd.DataFrame(
        {
            "row_id": studies.row_id,
            "study": studies.study,
            "mean_treat": studies.mean_treat,
            "sd_treat": studies.sd_treat,
            "n_treat": studies.n_treat,
            "mean_control": studies.mean_control,
            "sd_control": studies.sd_control,
            "n_control": studies.n_control,
            "effect": effects.effect,
            "effect_display": effects.effect,
            "variance": effects.variance,
            "standard_error": np.sqrt(effects.variance),
            "pooled_sd": effects.pooled_sd,
            "cohen_d": effects.cohen_d,
            "smd_correction_factor": effects.correction_factor,
            "included": included,
            "exclusion_reason": pd.Series(
                studies.exclusion_reason, dtype=object, copy=True
            ),
            "weight": raw_weights,
            "normalized_weight": normalized_weights,
        }
    )

    warnings = list(fit.warnings)
    excluded_count = int(np.count_nonzero(~included))
    if excluded_count:
        warnings.append(
            f"Excluded {excluded_count} study row(s) under missing={missing!r}."
        )

    options: tuple[tuple[str, MethodOptionValue], ...]
    if effects.measure == "SMD":
        options = (
            ("effect_estimator", "hedges_g_exact"),
            ("standardizer", "pooled_sd"),
            ("smd_variance", "LS"),
        )
    else:
        options = (("sampling_variance", "unpooled"),)
    method = MethodConfig(
        model=normalized_model,
        pooling_method="inverse_variance",
        tau2_method=None if normalized_model == "common" else normalized_tau2,
        ci_method=normalized_ci,
        confidence_level=confidence_level,
        prediction_interval_method="HTS" if normalized_model == "random" else None,
        missing=missing,
        options=options,
    )
    diagnostics = FitDiagnostics(
        converged=True if fit.tau2 is None else fit.tau2.converged,
        iterations=0 if fit.tau2 is None else fit.tau2.iterations,
        tau2_at_boundary=None if fit.tau2 is None else fit.tau2.boundary,
    )
    return MetaAnalysisResult(
        estimate=fit.estimate,
        standard_error=fit.standard_error,
        ci_low=fit.ci_low,
        ci_high=fit.ci_high,
        prediction_interval=fit.prediction_interval,
        tau2=0.0 if fit.tau2 is None else fit.tau2.value,
        heterogeneity=heterogeneity,
        k=len(included_effect),
        model=normalized_model,
        measure=effects.measure,
        effect_scale=effects.effect_scale,
        display_scale=effects.display_scale,
        method=method,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        _study_results=study_results,
    )


@overload
def meta_continuous(
    data: pd.DataFrame | None = None,
    *,
    mean_treat: ColumnOrArray,
    sd_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    mean_control: ColumnOrArray,
    sd_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: None = None,
    measure: str = "MD",
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    smd_variance: str = "LS",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult: ...


@overload
def meta_continuous(
    data: pd.DataFrame | None = None,
    *,
    mean_treat: ColumnOrArray,
    sd_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    mean_control: ColumnOrArray,
    sd_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray,
    measure: str = "MD",
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    smd_variance: str = "LS",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> SubgroupMetaAnalysisResult: ...


def meta_continuous(
    data: pd.DataFrame | None = None,
    *,
    mean_treat: ColumnOrArray,
    sd_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    mean_control: ColumnOrArray,
    sd_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray | None = None,
    measure: str = "MD",
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    smd_variance: str = "LS",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult | SubgroupMetaAnalysisResult:
    """Pool continuous outcomes, optionally fitting independent subgroups."""

    overall = _fit_meta_continuous_single(
        data,
        mean_treat=mean_treat,
        sd_treat=sd_treat,
        n_treat=n_treat,
        mean_control=mean_control,
        sd_control=sd_control,
        n_control=n_control,
        study=study,
        measure=measure,
        model=model,
        tau2_method=tau2_method,
        ci_method=ci_method,
        confidence_level=confidence_level,
        smd_variance=smd_variance,
        missing=missing,
        atol=atol,
        max_iter=max_iter,
    )
    if subgroup is None:
        return overall

    def fit_group(positions: np.ndarray) -> MetaAnalysisResult:
        rows = overall.study_results.iloc[positions]
        return _fit_meta_continuous_single(
            mean_treat=rows["mean_treat"].to_numpy(dtype=np.float64, copy=True),
            sd_treat=rows["sd_treat"].to_numpy(dtype=np.float64, copy=True),
            n_treat=rows["n_treat"].to_numpy(dtype=np.float64, copy=True),
            mean_control=rows["mean_control"].to_numpy(dtype=np.float64, copy=True),
            sd_control=rows["sd_control"].to_numpy(dtype=np.float64, copy=True),
            n_control=rows["n_control"].to_numpy(dtype=np.float64, copy=True),
            study=rows["study"].to_numpy(dtype=object, copy=True),
            measure=measure,
            model=model,
            tau2_method=tau2_method,
            ci_method=ci_method,
            confidence_level=confidence_level,
            smd_variance=smd_variance,
            missing=missing,
            atol=atol,
            max_iter=max_iter,
        )

    return fit_subgroup_analysis(
        data=data,
        subgroup=subgroup,
        overall=overall,
        fit_group=fit_group,
    )
