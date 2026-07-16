"""High-level public analysis functions."""

from __future__ import annotations

from dataclasses import replace
from typing import overload

import numpy as np
import pandas as pd

from .config import MethodConfig
from .data import ColumnOrArray, MissingPolicy, normalize_studies
from .estimators import fit_inverse_variance
from .exceptions import InvalidStudyDataError, UnsupportedMethodError
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


def _normalize_model(model: str) -> str:
    normalized = model.lower().replace("-", "_")
    if normalized in {"common", "common_effect", "fixed", "fixed_effect"}:
        return "common"
    if normalized in {"random", "random_effects", "random_effect"}:
        return "random"
    raise UnsupportedMethodError(
        f"Unsupported model={model!r}; expected 'common' or 'random'."
    )


def _normalize_ci_method(ci_method: str) -> str:
    normalized = ci_method.lower().replace("-", "_")
    aliases = {
        "classic": "normal",
        "z": "normal",
        "hksj": "hartung_knapp",
        "hk": "hartung_knapp",
        "adhoc": "hartung_knapp_adhoc",
        "hksj_adhoc": "hartung_knapp_adhoc",
    }
    return aliases.get(normalized, normalized)


def _validate_analysis_controls(
    *, confidence_level: float, atol: float, max_iter: int
) -> tuple[float, float, int]:
    if not isinstance(confidence_level, (int, float)) or not (
        0.0 < float(confidence_level) < 1.0
    ):
        raise InvalidStudyDataError("confidence_level must be between 0 and 1.")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise InvalidStudyDataError("max_iter must be a positive integer.")
    if not isinstance(atol, (int, float)) or not np.isfinite(atol) or atol <= 0.0:
        raise InvalidStudyDataError("atol must be finite and strictly positive.")
    return float(confidence_level), float(atol), max_iter


def _fit_meta_analysis_single(
    data: pd.DataFrame | None = None,
    *,
    effect: ColumnOrArray,
    variance: ColumnOrArray | None = None,
    standard_error: ColumnOrArray | None = None,
    study: ColumnOrArray | None = None,
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult:
    """Fit a generic inverse-variance meta-analysis.

    Parameters
    ----------
    data:
        Optional pandas DataFrame. String-valued input arguments select columns
        from this frame.
    effect:
        A DataFrame column name or one-dimensional array-like containing study
        effects.
    variance, standard_error:
        Exactly one must be provided as a DataFrame column name or
        one-dimensional array-like. Values must be finite and strictly
        positive. Standard errors are squared internally to obtain sampling
        variances.
    study:
        Optional study label column/array. DataFrame input defaults to its index;
        array-only input defaults to integer row labels.
    model:
        ``"common"`` (``"fixed"`` is an alias) or ``"random"``.
    tau2_method:
        ``"REML"`` (default), ``"PM"``, or ``"DL"`` for random-effects models.
    ci_method:
        ``"normal"``, ``"hartung_knapp"``, or
        ``"hartung_knapp_adhoc"``.
    confidence_level:
        Confidence level strictly between zero and one.
    missing:
        ``"raise"`` (default) or ``"drop"``. Dropped studies remain visible in
        the result with a structured exclusion reason.
    atol, max_iter:
        Numerical controls for iterative tau-squared estimators.
    """

    confidence_level, atol, max_iter = _validate_analysis_controls(
        confidence_level=confidence_level,
        atol=atol,
        max_iter=max_iter,
    )

    normalized_model = _normalize_model(model)
    normalized_ci = _normalize_ci_method(ci_method)
    normalized_tau2 = tau2_method.upper().replace("-", "_")

    studies = normalize_studies(
        data=data,
        effect=effect,
        variance=variance,
        standard_error=standard_error,
        study=study,
        missing=missing,
    )
    included_effect = studies.included_effect
    included_variance = studies.included_variance

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
    tau2_value = 0.0 if fit.tau2 is None else fit.tau2.value
    i2_method = "q_based"
    if normalized_model == "random":
        i2, h2 = tau2_inconsistency(included_variance, tau2_value)
        i2_method = "tau2_typical_variance"
    heterogeneity = HeterogeneityResult(q, q_df, q_pvalue, i2, h2, i2_method)

    row_count = len(studies.row_id)
    raw_weights = np.full(row_count, np.nan, dtype=np.float64)
    normalized_weights = np.full(row_count, np.nan, dtype=np.float64)
    raw_weights[studies.included] = fit.weights
    normalized_weights[studies.included] = fit.normalized_weights
    study_results = pd.DataFrame(
        {
            "row_id": studies.row_id,
            "study": studies.study,
            "effect": studies.effect,
            "variance": studies.variance,
            "standard_error": np.sqrt(studies.variance),
            "included": studies.included,
            "exclusion_reason": pd.Series(
                studies.exclusion_reason, dtype=object, copy=True
            ),
            "weight": raw_weights,
            "normalized_weight": normalized_weights,
        }
    )

    warnings = list(fit.warnings)
    excluded_count = int(np.count_nonzero(~studies.included))
    if excluded_count:
        warnings.append(
            f"Excluded {excluded_count} study row(s) under missing={missing!r}."
        )

    diagnostics = FitDiagnostics(
        converged=True if fit.tau2 is None else fit.tau2.converged,
        iterations=0 if fit.tau2 is None else fit.tau2.iterations,
        tau2_at_boundary=None if fit.tau2 is None else fit.tau2.boundary,
    )
    method = MethodConfig(
        model=normalized_model,
        pooling_method="inverse_variance",
        tau2_method=None if normalized_model == "common" else normalized_tau2,
        ci_method=normalized_ci,
        confidence_level=confidence_level,
        prediction_interval_method="HTS" if normalized_model == "random" else None,
        missing=missing,
        atol=atol,
        max_iter=max_iter,
        options=(),
    )
    transformations: tuple[TransformationRecord, ...] = ()
    if standard_error is not None:
        uncertainty_input = ("standard_error", standard_error)
        transformed_rows = tuple(
            int(row) for row in np.flatnonzero(~pd.isna(studies.variance))
        )
        transformations = (
            TransformationRecord(
                name="standard_error_to_variance",
                affected_rows=transformed_rows,
            ),
        )
    else:
        if variance is None:  # pragma: no cover - validated by normalize_studies
            raise RuntimeError("variance input unexpectedly missing")
        uncertainty_input = ("variance", variance)

    provenance = build_analysis_provenance(
        analysis_type="generic",
        data=data,
        inputs=(("effect", effect), uncertainty_input),
        study=study,
        included=studies.included,
        transformations=transformations,
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
        measure="GENERIC",
        effect_scale="identity",
        display_scale="identity",
        method=method,
        diagnostics=diagnostics,
        provenance=provenance,
        warnings=tuple(warnings),
        _study_results=study_results,
        _source_data=data,
    )


@overload
def meta_analysis(
    data: pd.DataFrame | None = None,
    *,
    effect: ColumnOrArray,
    variance: ColumnOrArray,
    standard_error: None = None,
    study: ColumnOrArray | None = None,
    subgroup: None = None,
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult: ...


@overload
def meta_analysis(
    data: pd.DataFrame | None = None,
    *,
    effect: ColumnOrArray,
    variance: None = None,
    standard_error: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: None = None,
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult: ...


@overload
def meta_analysis(
    data: pd.DataFrame | None = None,
    *,
    effect: ColumnOrArray,
    variance: ColumnOrArray,
    standard_error: None = None,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray,
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> SubgroupMetaAnalysisResult: ...


@overload
def meta_analysis(
    data: pd.DataFrame | None = None,
    *,
    effect: ColumnOrArray,
    variance: None = None,
    standard_error: ColumnOrArray,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray,
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> SubgroupMetaAnalysisResult: ...


def meta_analysis(
    data: pd.DataFrame | None = None,
    *,
    effect: ColumnOrArray,
    variance: ColumnOrArray | None = None,
    standard_error: ColumnOrArray | None = None,
    study: ColumnOrArray | None = None,
    subgroup: ColumnOrArray | None = None,
    model: str = "random",
    tau2_method: str = "REML",
    ci_method: str = "normal",
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaAnalysisResult | SubgroupMetaAnalysisResult:
    """Fit a generic inverse-variance meta-analysis, optionally by subgroup.

    ``effect`` and the selected uncertainty input accept DataFrame column names
    or one-dimensional array-like values. Supply exactly one of ``variance`` or
    ``standard_error``; standard errors are squared internally. Uncertainty
    values must be finite and strictly positive. The default is a REML
    random-effects model with a normal confidence interval. ``subgroup``
    returns :class:`SubgroupMetaAnalysisResult` when supplied; otherwise the
    return value is :class:`MetaAnalysisResult`. Missing subgroup labels are
    rejected explicitly.
    """

    overall = _fit_meta_analysis_single(
        data,
        effect=effect,
        variance=variance,
        standard_error=standard_error,
        study=study,
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

    def fit_group(positions: np.ndarray) -> MetaAnalysisResult:
        rows = overall.study_results.iloc[positions]
        return _fit_meta_analysis_single(
            effect=rows["effect"].to_numpy(dtype=np.float64, copy=True),
            variance=rows["variance"].to_numpy(dtype=np.float64, copy=True),
            study=rows["study"].to_numpy(dtype=object, copy=True),
            model=model,
            tau2_method=tau2_method,
            ci_method=ci_method,
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
