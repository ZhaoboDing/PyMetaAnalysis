"""High-level API for two-group binary outcome meta-analysis."""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

from .api import (
    _normalize_ci_method,
    _normalize_model,
    _validate_analysis_controls,
)
from .config import MethodConfig
from .data import ColumnOrArray, MissingPolicy
from .effect_sizes.binary import (
    adjusted_tables,
    calculate_binary_effects,
    normalize_binary_studies,
    normalize_correction_scope,
    validate_correction,
)
from .estimators import fit_inverse_variance, fit_mantel_haenszel
from .exceptions import UnsupportedMethodError
from .heterogeneity import classical_heterogeneity, heterogeneity_at_estimate
from .results import (
    FitDiagnostics,
    HeterogeneityResult,
    MetaAnalysisResult,
    SubgroupMetaAnalysisResult,
)
from .subgroups import fit_subgroup_analysis


def _normalize_pooling_method(method: str) -> str:
    normalized = method.lower().replace("-", "_")
    if normalized in {"iv", "inverse", "inverse_variance"}:
        return "inverse_variance"
    if normalized in {"mh", "mantel_haenszel"}:
        return "mantel_haenszel"
    raise UnsupportedMethodError(
        "method must be 'MH'/'mantel_haenszel' or 'IV'/'inverse_variance'."
    )


def _fit_meta_binary_single(
    data: pd.DataFrame | None = None,
    *,
    event_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    event_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None = None,
    measure: str = "RR",
    method: str = "MH",
    model: str = "common",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    continuity_correction: float = 0.5,
    correction_scope: str = "only_zero_studies",
    mh_continuity_correction: float | None = None,
    mh_correction_scope: str = "only_zero_studies",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult:
    """Pool OR, RR, or RD from two-group binary study counts.

    Mantel-Haenszel currently supports common-effect OR and RR. It uses raw
    tables by default; set ``mh_continuity_correction`` explicitly when the
    exact pooled estimator is undefined. Study-level effects use the separate
    ``continuity_correction`` setting for display and heterogeneity statistics.
    """

    confidence_level, atol, max_iter = _validate_analysis_controls(
        confidence_level=confidence_level,
        atol=atol,
        max_iter=max_iter,
    )
    normalized_model = _normalize_model(model)
    normalized_method = _normalize_pooling_method(method)
    normalized_ci = _normalize_ci_method(ci_method)
    normalized_tau2 = tau2_method.upper().replace("-", "_")
    normalized_measure = measure.upper()
    correction = validate_correction(
        continuity_correction, name="continuity_correction"
    )
    mh_correction = validate_correction(
        mh_continuity_correction, name="mh_continuity_correction"
    )
    scope = normalize_correction_scope(correction_scope)
    mh_scope = normalize_correction_scope(mh_correction_scope)

    if normalized_method == "mantel_haenszel":
        if normalized_model != "common":
            raise UnsupportedMethodError(
                "Mantel-Haenszel is currently implemented only for model='common'; "
                "use method='IV' for random-effects models."
            )
        if normalized_measure not in {"OR", "RR"}:
            raise UnsupportedMethodError(
                "Mantel-Haenszel currently supports measure='OR' or measure='RR'; "
                "use method='IV' for risk differences."
            )
        if normalized_ci != "normal":
            raise UnsupportedMethodError(
                "Mantel-Haenszel currently supports only ci_method='normal'."
            )

    studies = normalize_binary_studies(
        data=data,
        event_treat=event_treat,
        n_treat=n_treat,
        event_control=event_control,
        n_control=n_control,
        study=study,
        missing=missing,
    )
    effects = calculate_binary_effects(
        studies,
        measure=normalized_measure,
        continuity_correction=correction,
        correction_scope=scope,
    )
    included = effects.studies.included
    included_effect = effects.included_effect
    included_variance = effects.included_variance

    warnings: list[str] = []
    if normalized_method == "inverse_variance":
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
        estimate = fit.estimate
        standard_error = fit.standard_error
        ci_low = fit.ci_low
        ci_high = fit.ci_high
        prediction_interval = fit.prediction_interval
        weights = fit.weights
        normalized_weights = fit.normalized_weights
        tau2 = 0.0 if fit.tau2 is None else fit.tau2.value
        diagnostics = FitDiagnostics(
            converged=True if fit.tau2 is None else fit.tau2.converged,
            iterations=0 if fit.tau2 is None else fit.tau2.iterations,
            tau2_at_boundary=None if fit.tau2 is None else fit.tau2.boundary,
        )
        q_values = classical_heterogeneity(included_effect, included_variance)
        mh_corrected = np.zeros(len(included), dtype=bool)
        warnings.extend(fit.warnings)
    else:
        a, b, c, d, mh_corrected = adjusted_tables(
            effects.studies,
            correction=mh_correction,
            scope=mh_scope,
        )
        mh_fit = fit_mantel_haenszel(
            a[included],
            b[included],
            c[included],
            d[included],
            measure=normalized_measure,
            confidence_level=confidence_level,
        )
        estimate = mh_fit.estimate
        standard_error = mh_fit.standard_error
        ci_low = mh_fit.ci_low
        ci_high = mh_fit.ci_high
        prediction_interval = None
        weights = mh_fit.weights
        normalized_weights = mh_fit.normalized_weights
        tau2 = 0.0
        diagnostics = FitDiagnostics(True, 0, None)
        q_values = heterogeneity_at_estimate(
            included_effect, included_variance, estimate
        )

    q, q_df, q_pvalue, i2, h2 = q_values
    heterogeneity = HeterogeneityResult(q, q_df, q_pvalue, i2, h2)
    row_count = len(included)
    raw_weights = np.full(row_count, np.nan, dtype=np.float64)
    result_weights = np.full(row_count, np.nan, dtype=np.float64)
    raw_weights[included] = weights
    result_weights[included] = normalized_weights
    effect_display = effects.effect.copy()
    if effects.display_scale == "exp":
        effect_display[included] = np.exp(effect_display[included])

    study_results = pd.DataFrame(
        {
            "row_id": effects.studies.row_id,
            "study": effects.studies.study,
            "event_treat": effects.studies.event_treat,
            "n_treat": effects.studies.n_treat,
            "event_control": effects.studies.event_control,
            "n_control": effects.studies.n_control,
            "effect": effects.effect,
            "effect_display": effect_display,
            "variance": effects.variance,
            "standard_error": np.sqrt(effects.variance),
            "included": included,
            "exclusion_reason": pd.Series(
                effects.studies.exclusion_reason, dtype=object, copy=True
            ),
            "continuity_corrected": effects.corrected,
            "mh_continuity_corrected": mh_corrected,
            "weight": raw_weights,
            "normalized_weight": result_weights,
        }
    )

    excluded_count = int(np.count_nonzero(~included))
    corrected_count = int(np.count_nonzero(effects.corrected))
    mh_corrected_count = int(np.count_nonzero(mh_corrected))
    if excluded_count:
        warnings.append(
            f"Excluded {excluded_count} non-informative or missing study row(s)."
        )
    if corrected_count:
        warnings.append(
            f"Applied continuity_correction={correction:g} to "
            f"{corrected_count} study table(s) for individual effects."
        )
    if mh_corrected_count:
        warnings.append(
            f"Applied mh_continuity_correction={mh_correction:g} to "
            f"{mh_corrected_count} study table(s) for MH pooling."
        )

    method_config = MethodConfig(
        model=normalized_model,
        pooling_method=normalized_method,
        tau2_method=(
            normalized_tau2
            if normalized_model == "random" and normalized_method == "inverse_variance"
            else None
        ),
        ci_method=normalized_ci,
        confidence_level=confidence_level,
        prediction_interval_method=(
            "HTS"
            if normalized_model == "random" and normalized_method == "inverse_variance"
            else None
        ),
        missing=missing,
        atol=atol,
        max_iter=max_iter,
        options=(
            ("continuity_correction", correction),
            ("correction_scope", scope),
            ("mh_continuity_correction", mh_correction),
            ("mh_correction_scope", mh_scope),
        ),
    )
    return MetaAnalysisResult(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        prediction_interval=prediction_interval,
        tau2=tau2,
        heterogeneity=heterogeneity,
        k=len(included_effect),
        model=normalized_model,
        measure=normalized_measure,
        effect_scale=effects.effect_scale,
        display_scale=effects.display_scale,
        method=method_config,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        _study_results=study_results,
        _source_data=data,
    )


@overload
def meta_binary(
    data: pd.DataFrame | None = None,
    *,
    event_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    event_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: None = None,
    measure: str = "RR",
    method: str = "MH",
    model: str = "common",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    continuity_correction: float = 0.5,
    correction_scope: str = "only_zero_studies",
    mh_continuity_correction: float | None = None,
    mh_correction_scope: str = "only_zero_studies",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult: ...


@overload
def meta_binary(
    data: pd.DataFrame | None = None,
    *,
    event_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    event_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray,
    measure: str = "RR",
    method: str = "MH",
    model: str = "common",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    continuity_correction: float = 0.5,
    correction_scope: str = "only_zero_studies",
    mh_continuity_correction: float | None = None,
    mh_correction_scope: str = "only_zero_studies",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> SubgroupMetaAnalysisResult: ...


def meta_binary(
    data: pd.DataFrame | None = None,
    *,
    event_treat: ColumnOrArray,
    n_treat: ColumnOrArray,
    event_control: ColumnOrArray,
    n_control: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray | None = None,
    measure: str = "RR",
    method: str = "MH",
    model: str = "common",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    continuity_correction: float = 0.5,
    correction_scope: str = "only_zero_studies",
    mh_continuity_correction: float | None = None,
    mh_correction_scope: str = "only_zero_studies",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult | SubgroupMetaAnalysisResult:
    """Pool binary outcomes, optionally fitting independent study subgroups."""

    overall = _fit_meta_binary_single(
        data,
        event_treat=event_treat,
        n_treat=n_treat,
        event_control=event_control,
        n_control=n_control,
        study=study,
        measure=measure,
        method=method,
        model=model,
        tau2_method=tau2_method,
        ci_method=ci_method,
        confidence_level=confidence_level,
        continuity_correction=continuity_correction,
        correction_scope=correction_scope,
        mh_continuity_correction=mh_continuity_correction,
        mh_correction_scope=mh_correction_scope,
        missing=missing,
        atol=atol,
        max_iter=max_iter,
    )
    if subgroup is None:
        return overall

    def fit_group(positions: np.ndarray) -> MetaAnalysisResult:
        rows = overall.study_results.iloc[positions]
        return _fit_meta_binary_single(
            event_treat=rows["event_treat"].to_numpy(dtype=np.float64, copy=True),
            n_treat=rows["n_treat"].to_numpy(dtype=np.float64, copy=True),
            event_control=rows["event_control"].to_numpy(dtype=np.float64, copy=True),
            n_control=rows["n_control"].to_numpy(dtype=np.float64, copy=True),
            study=rows["study"].to_numpy(dtype=object, copy=True),
            measure=measure,
            method=method,
            model=model,
            tau2_method=tau2_method,
            ci_method=ci_method,
            confidence_level=confidence_level,
            continuity_correction=continuity_correction,
            correction_scope=correction_scope,
            mh_continuity_correction=mh_continuity_correction,
            mh_correction_scope=mh_correction_scope,
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
