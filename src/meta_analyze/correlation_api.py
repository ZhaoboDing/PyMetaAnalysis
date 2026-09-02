"""High-level API for meta-analysis of independent correlations."""

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
from .config import MethodConfig, MethodOptionValue
from .data import ColumnOrArray, MissingPolicy, _duplicate_study_warning
from .effect_sizes.correlation import (
    calculate_correlation_effects,
    normalize_correlation_studies,
)
from .estimators import fit_inverse_variance
from .heterogeneity import classical_heterogeneity, tau2_inconsistency
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


def _fit_meta_correlation_single(
    data: pd.DataFrame | None = None,
    *,
    correlation: ColumnOrArray,
    n: ColumnOrArray,
    study: ColumnOrArray | None = None,
    measure: str = "ZCOR",
    model: str = "random",
    tau2_method: str | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult:
    """Pool correlations after Fisher's r-to-z transformation."""

    confidence_level, atol, max_iter = _validate_analysis_controls(
        confidence_level=confidence_level,
        atol=atol,
        max_iter=max_iter,
    )
    normalized_model = _normalize_model(model)
    normalized_ci = _normalize_ci_method(ci_method)
    normalized_tau2 = _resolve_tau2_method(
        tau2_method,
        applicable=normalized_model == "random",
    )

    studies = normalize_correlation_studies(
        data=data,
        correlation=correlation,
        n=n,
        study=study,
        missing=missing,
    )
    effects = calculate_correlation_effects(studies, measure=measure)
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
        included_effect,
        included_variance,
    )
    tau2_value = 0.0 if fit.tau2 is None else fit.tau2.value
    i2_method = "q_based"
    if normalized_model == "random":
        i2, h2 = tau2_inconsistency(included_variance, tau2_value)
        i2_method = "tau2_typical_variance"
    heterogeneity = HeterogeneityResult(q, q_df, q_pvalue, i2, h2, i2_method)

    row_count = len(included)
    raw_weights = np.full(row_count, np.nan, dtype=np.float64)
    normalized_weights = np.full(row_count, np.nan, dtype=np.float64)
    raw_weights[included] = fit.weights
    normalized_weights[included] = fit.normalized_weights
    study_results = pd.DataFrame(
        {
            "row_id": studies.row_id,
            "study": studies.study,
            "correlation": studies.correlation,
            "n": studies.n,
            "effect": effects.effect,
            "effect_display": studies.correlation,
            "variance": effects.variance,
            "standard_error": np.sqrt(effects.variance),
            "included": included,
            "exclusion_reason": pd.Series(
                studies.exclusion_reason,
                dtype=object,
                copy=True,
            ),
            "weight": raw_weights,
            "normalized_weight": normalized_weights,
        }
    )

    warnings = list(fit.warnings)
    duplicate_warning = _duplicate_study_warning(studies.study)
    if duplicate_warning is not None:
        warnings.append(duplicate_warning)
    excluded_count = int(np.count_nonzero(~included))
    if excluded_count:
        warnings.append(
            f"Excluded {excluded_count} study row(s) under missing={missing!r}."
        )

    options: tuple[tuple[str, MethodOptionValue], ...] = (
        ("effect_transformation", "fisher_r_to_z"),
        ("sampling_variance", "1/(n-3)"),
        ("display_transformation", "tanh"),
    )
    method = MethodConfig(
        model=normalized_model,
        pooling_method="inverse_variance",
        tau2_method=None if normalized_model == "common" else normalized_tau2,
        ci_method=normalized_ci,
        confidence_level=confidence_level,
        prediction_interval_method=_prediction_interval_method(
            normalized_model,
            normalized_ci,
            available=fit.prediction_interval is not None,
        ),
        missing=missing,
        atol=atol,
        max_iter=max_iter,
        options=options,
    )
    diagnostics = FitDiagnostics(
        converged=True if fit.tau2 is None else fit.tau2.converged,
        iterations=0 if fit.tau2 is None else fit.tau2.iterations,
        tau2_at_boundary=None if fit.tau2 is None else fit.tau2.boundary,
    )
    provenance = build_analysis_provenance(
        analysis_type="correlation",
        data=data,
        inputs=(("correlation", correlation), ("n", n)),
        study=study,
        included=included,
        transformations=(
            TransformationRecord(
                name="fisher_r_to_z",
                parameters=(
                    ("measure", effects.measure),
                    ("sampling_variance", "1/(n-3)"),
                    ("display_transformation", "tanh"),
                ),
                affected_rows=tuple(int(row) for row in np.flatnonzero(included)),
            ),
        ),
    )
    return MetaAnalysisResult(
        estimate=fit.estimate,
        standard_error=fit.standard_error,
        ci_low=fit.ci_low,
        ci_high=fit.ci_high,
        prediction_interval=fit.prediction_interval,
        tau2=tau2_value,
        heterogeneity=heterogeneity,
        k=len(included_effect),
        model=normalized_model,
        measure=effects.measure,
        effect_scale=effects.effect_scale,
        display_scale=effects.display_scale,
        method=method,
        diagnostics=diagnostics,
        provenance=provenance,
        warnings=tuple(warnings),
        _study_results=study_results,
        _source_data=data,
    )


@overload
def meta_correlation(
    data: pd.DataFrame | None = None,
    *,
    correlation: ColumnOrArray,
    n: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: None = None,
    measure: str = "ZCOR",
    model: str = "random",
    tau2_method: str | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult: ...


@overload
def meta_correlation(
    data: pd.DataFrame | None = None,
    *,
    correlation: ColumnOrArray,
    n: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray,
    measure: str = "ZCOR",
    model: str = "random",
    tau2_method: str | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> SubgroupMetaAnalysisResult: ...


def meta_correlation(
    data: pd.DataFrame | None = None,
    *,
    correlation: ColumnOrArray,
    n: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray | None = None,
    measure: str = "ZCOR",
    model: str = "random",
    tau2_method: str | None = None,
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult | SubgroupMetaAnalysisResult:
    """Pool independent correlations, optionally by subgroup.

    Correlations and sample sizes accept DataFrame column names or
    one-dimensional array-like values. ``measure="ZCOR"`` applies Fisher's
    r-to-z transformation and sampling variance ``1 / (n - 3)``. Models are
    fitted on the z scale and displayed as back-transformed correlations.
    """

    overall = _fit_meta_correlation_single(
        data,
        correlation=correlation,
        n=n,
        study=study,
        measure=measure,
        model=model,
        tau2_method=tau2_method,
        ci_method=ci_method,
        confidence_level=confidence_level,
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
        return _fit_meta_correlation_single(
            correlation=rows["correlation"].to_numpy(dtype=np.float64, copy=True),
            n=rows["n"].to_numpy(dtype=np.float64, copy=True),
            study=rows["study"].to_numpy(dtype=object, copy=True),
            measure=overall.measure,
            model="common" if singleton_random else model,
            tau2_method=None if singleton_random else tau2_method,
            ci_method="normal" if singleton_random else ci_method,
            confidence_level=confidence_level,
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
