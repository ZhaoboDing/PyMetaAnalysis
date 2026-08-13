"""High-level API for two-group binary outcome meta-analysis."""

from __future__ import annotations

from dataclasses import replace
from typing import overload

import numpy as np
import pandas as pd

from .api import (
    _normalize_ci_method,
    _normalize_model,
    _prediction_interval_method,
    _resolve_tau2_method,
    _validate_analysis_controls,
)
from .config import MethodConfig
from .data import ColumnOrArray, MissingPolicy, _duplicate_study_warning
from .effect_sizes.binary import (
    adjusted_tables,
    calculate_binary_effects,
    calculate_peto_effects,
    normalize_binary_studies,
    normalize_correction_scope,
    normalize_rd_zero_variance,
    validate_correction,
)
from .estimators import fit_inverse_variance, fit_mantel_haenszel, fit_peto
from .exceptions import UnsupportedMethodError
from .heterogeneity import (
    classical_heterogeneity,
    heterogeneity_at_estimate,
    tau2_inconsistency,
)
from .provenance import (
    TransformationRecord,
    add_input_field,
    build_analysis_provenance,
)
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
    if normalized in {"peto", "peto_one_step"}:
        return "peto"
    raise UnsupportedMethodError(
        "method must be 'MH'/'mantel_haenszel', 'Peto'/'peto_one_step', "
        "or 'IV'/'inverse_variance'."
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
    tau2_method: str | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    continuity_correction: float = 0.5,
    correction_scope: str = "only_zero_studies",
    rd_zero_variance: str = "correct",
    mh_continuity_correction: float | None = None,
    mh_correction_scope: str = "only_zero_studies",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult:
    """Pool OR, RR, or RD from two-group binary study counts.

    Mantel-Haenszel supports common-effect OR, RR, and RD. It uses raw tables
    by default; set ``mh_continuity_correction`` explicitly when the pooled
    estimator or variance is undefined. Study-level effects use the separate
    ``continuity_correction`` setting for display and heterogeneity statistics.
    Peto supports common-effect OR, uses raw tables for pooling, and applies
    the study-level correction only to displayed study estimates.
    ``tau2_method=None`` selects REML only when a random-effects
    inverse-variance model is requested; an explicit tau-squared method is
    rejected for common-effect and specialized table-based fits.
    """

    confidence_level, atol, max_iter = _validate_analysis_controls(
        confidence_level=confidence_level,
        atol=atol,
        max_iter=max_iter,
    )
    normalized_model = _normalize_model(model)
    normalized_method = _normalize_pooling_method(method)
    normalized_ci = _normalize_ci_method(ci_method)
    normalized_tau2 = _resolve_tau2_method(
        tau2_method,
        applicable=normalized_model == "random",
    )
    normalized_measure = measure.upper()
    correction = validate_correction(
        continuity_correction, name="continuity_correction"
    )
    mh_correction = validate_correction(
        mh_continuity_correction, name="mh_continuity_correction"
    )
    scope = normalize_correction_scope(correction_scope)
    rd_policy = normalize_rd_zero_variance(rd_zero_variance)
    mh_scope = normalize_correction_scope(mh_correction_scope)

    if normalized_measure != "RD" and rd_policy != "correct":
        raise UnsupportedMethodError(
            "rd_zero_variance is only configurable when measure='RD'."
        )

    if normalized_method == "mantel_haenszel":
        if normalized_model != "common":
            raise UnsupportedMethodError(
                "Mantel-Haenszel is currently implemented only for model='common'; "
                "use method='IV' for random-effects models."
            )
        if normalized_measure not in {"OR", "RR", "RD"}:
            raise UnsupportedMethodError(
                "Mantel-Haenszel supports measure='OR', measure='RR', or measure='RD'."
            )
        if normalized_ci != "normal":
            raise UnsupportedMethodError(
                "Mantel-Haenszel currently supports only ci_method='normal'."
            )
    elif normalized_method == "peto":
        if normalized_model != "common":
            raise UnsupportedMethodError(
                "Peto is implemented only for model='common'; use method='IV' "
                "for random-effects models."
            )
        if normalized_measure != "OR":
            raise UnsupportedMethodError("Peto pooling supports only measure='OR'.")
        if normalized_ci != "normal":
            raise UnsupportedMethodError(
                "Peto pooling supports only ci_method='normal'."
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
    effects = (
        calculate_peto_effects(
            studies,
            continuity_correction=correction,
            correction_scope=scope,
        )
        if normalized_method == "peto"
        else calculate_binary_effects(
            studies,
            measure=normalized_measure,
            continuity_correction=correction,
            correction_scope=scope,
            rd_zero_variance=rd_policy,
        )
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
    elif normalized_method == "mantel_haenszel":
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
    else:
        a, b, c, d, mh_corrected = adjusted_tables(
            effects.studies,
            correction=0.0,
            scope="none",
        )
        peto_fit = fit_peto(
            a[included],
            b[included],
            c[included],
            d[included],
            confidence_level=confidence_level,
        )
        estimate = peto_fit.estimate
        standard_error = peto_fit.standard_error
        ci_low = peto_fit.ci_low
        ci_high = peto_fit.ci_high
        prediction_interval = None
        weights = peto_fit.weights
        normalized_weights = peto_fit.normalized_weights
        tau2 = 0.0
        diagnostics = FitDiagnostics(True, 0, None)
        q_values = (
            peto_fit.q,
            peto_fit.q_df,
            peto_fit.q_pvalue,
            peto_fit.i2,
            peto_fit.h2,
        )
        warnings.append(
            "Peto's odds-ratio approximation is intended for rare outcomes, "
            "similar treatment/control group sizes within studies, and effects "
            "that are not large."
        )

    q, q_df, q_pvalue, i2, h2 = q_values
    i2_method = "q_based"
    if normalized_model == "random":
        i2, h2 = tau2_inconsistency(included_variance, tau2)
        i2_method = "tau2_typical_variance"
    heterogeneity = HeterogeneityResult(q, q_df, q_pvalue, i2, h2, i2_method)
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
            "rd_zero_variance": effects.rd_zero_variance,
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
    duplicate_warning = _duplicate_study_warning(effects.studies.study)
    if duplicate_warning is not None:
        warnings.append(duplicate_warning)

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
            _prediction_interval_method(
                normalized_model,
                normalized_ci,
                available=fit.prediction_interval is not None,
            )
            if normalized_method == "inverse_variance"
            else None
        ),
        missing=missing,
        atol=atol,
        max_iter=max_iter,
        options=(
            ("continuity_correction", correction),
            ("correction_scope", scope),
            *((("rd_zero_variance", rd_policy),) if normalized_measure == "RD" else ()),
            *(
                (("mh_rd_variance", "Sato-Greenland-Robins"),)
                if normalized_measure == "RD" and normalized_method == "mantel_haenszel"
                else ()
            ),
            *(
                (
                    ("peto_pooling_tables", "raw"),
                    ("peto_heterogeneity", "O-minus-E"),
                )
                if normalized_method == "peto"
                else ()
            ),
            ("mh_continuity_correction", mh_correction),
            ("mh_correction_scope", mh_scope),
        ),
    )
    transformations = [
        TransformationRecord(
            name="binary_effect_size",
            parameters=(
                ("measure", normalized_measure),
                ("model_scale", effects.effect_scale),
                ("display_scale", effects.display_scale),
                *(
                    (("study_estimator", "peto_one_step"),)
                    if normalized_method == "peto"
                    else ()
                ),
            ),
            affected_rows=tuple(int(row) for row in np.flatnonzero(included)),
        ),
        TransformationRecord(
            name="continuity_correction",
            parameters=(
                ("value", correction),
                ("scope", scope),
                ("target", "individual_effects"),
            ),
            affected_rows=tuple(int(row) for row in np.flatnonzero(effects.corrected)),
        ),
    ]
    if normalized_measure == "RD":
        transformations.append(
            TransformationRecord(
                name="rd_zero_variance_policy",
                parameters=(
                    ("policy", rd_policy),
                    ("variance_correction", correction),
                ),
                affected_rows=tuple(
                    int(row) for row in np.flatnonzero(effects.rd_zero_variance)
                ),
            )
        )
    relative_exclusions = np.flatnonzero(
        (~included)
        & np.isin(
            effects.studies.exclusion_reason,
            [
                "no events in either group",
                "all participants have events in both groups",
            ],
        )
    )
    if len(relative_exclusions):
        transformations.append(
            TransformationRecord(
                name="relative_effect_exclusion",
                parameters=(("measure", normalized_measure),),
                affected_rows=tuple(int(row) for row in relative_exclusions),
            )
        )
    if normalized_method == "mantel_haenszel":
        transformations.append(
            TransformationRecord(
                name="mantel_haenszel_continuity_correction",
                parameters=(
                    ("value", mh_correction),
                    ("scope", mh_scope),
                    ("target", "pooling"),
                ),
                affected_rows=tuple(int(row) for row in np.flatnonzero(mh_corrected)),
            )
        )
    provenance = build_analysis_provenance(
        analysis_type="binary",
        data=data,
        inputs=(
            ("event_treat", event_treat),
            ("n_treat", n_treat),
            ("event_control", event_control),
            ("n_control", n_control),
        ),
        study=study,
        included=included,
        transformations=tuple(transformations),
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
        provenance=provenance,
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
    tau2_method: str | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    continuity_correction: float = 0.5,
    correction_scope: str = "only_zero_studies",
    rd_zero_variance: str = "correct",
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
    tau2_method: str | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    continuity_correction: float = 0.5,
    correction_scope: str = "only_zero_studies",
    rd_zero_variance: str = "correct",
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
    tau2_method: str | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    continuity_correction: float = 0.5,
    correction_scope: str = "only_zero_studies",
    rd_zero_variance: str = "correct",
    mh_continuity_correction: float | None = None,
    mh_correction_scope: str = "only_zero_studies",
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult | SubgroupMetaAnalysisResult:
    """Pool binary outcomes, optionally fitting independent study subgroups.

    Event and total arguments accept DataFrame column names or one-dimensional
    array-like values. The default is common-effect Mantel-Haenszel risk-ratio
    pooling. Mantel-Haenszel OR, RR, and RD and Peto OR are available for
    common-effect models; use inverse-variance pooling for random effects. For risk
    differences, ``rd_zero_variance="correct"`` retains boundary studies with
    their raw effect and corrected study-level sampling variance. Use
    ``rd_zero_variance="exclude"`` to remove them before all synthesis
    calculations.
    """

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
        rd_zero_variance=rd_zero_variance,
        mh_continuity_correction=mh_continuity_correction,
        mh_correction_scope=mh_correction_scope,
        missing=missing,
        atol=atol,
        max_iter=max_iter,
    )
    if subgroup is None:
        return overall

    overall = replace(
        overall,
        provenance=add_input_field(
            overall.provenance,
            role="subgroup",
            value=subgroup,
            data=data,
        ),
    )

    def fit_group(
        positions: np.ndarray,
        singleton_random: bool,
    ) -> MetaAnalysisResult:
        rows = overall.study_results.iloc[positions]
        return _fit_meta_binary_single(
            event_treat=rows["event_treat"].to_numpy(dtype=np.float64, copy=True),
            n_treat=rows["n_treat"].to_numpy(dtype=np.float64, copy=True),
            event_control=rows["event_control"].to_numpy(dtype=np.float64, copy=True),
            n_control=rows["n_control"].to_numpy(dtype=np.float64, copy=True),
            study=rows["study"].to_numpy(dtype=object, copy=True),
            measure=measure,
            method=method,
            model="common" if singleton_random else model,
            tau2_method=None if singleton_random else tau2_method,
            ci_method="normal" if singleton_random else ci_method,
            confidence_level=confidence_level,
            continuity_correction=continuity_correction,
            correction_scope=correction_scope,
            rd_zero_variance=rd_zero_variance,
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
