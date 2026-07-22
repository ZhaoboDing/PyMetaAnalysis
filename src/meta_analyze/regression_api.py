"""Public pandas-first meta-regression API."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .api import _validate_analysis_controls
from .config import MetaRegressionMethodConfig
from .data import ColumnOrArray, MissingPolicy
from .design_matrix import (
    CategoricalInput,
    ModeratorInput,
    normalize_meta_regression_data,
)
from .estimators import (
    estimate_meta_regression_tau2,
    fit_meta_regression,
    residual_heterogeneity,
)
from .exceptions import UnsupportedMethodError
from .provenance import TransformationRecord, build_analysis_provenance
from .regression_results import (
    MetaRegressionDiagnostics,
    MetaRegressionResult,
    ModeratorTestResult,
)
from .results import HeterogeneityResult


def _normalize_regression_model(model: str) -> str:
    normalized = model.lower().replace("-", "_")
    if normalized in {"common", "common_effect", "fixed", "fixed_effect"}:
        return "common"
    if normalized in {
        "mixed",
        "mixed_effects",
        "mixed_effect",
        "random",
        "random_effects",
        "random_effect",
    }:
        return "mixed"
    raise UnsupportedMethodError(
        f"Unsupported model={model!r}; expected 'common' or 'mixed'."
    )


def _normalize_inference_method(inference_method: str) -> str:
    normalized = inference_method.lower().replace("-", "_")
    aliases = {
        "classic": "normal",
        "z": "normal",
        "hksj": "hartung_knapp",
        "hk": "hartung_knapp",
        "adhoc": "hartung_knapp_adhoc",
        "hksj_adhoc": "hartung_knapp_adhoc",
    }
    return aliases.get(normalized, normalized)


def _moderator_inputs(
    moderators: ModeratorInput,
) -> tuple[tuple[str, ColumnOrArray], ...]:
    if isinstance(moderators, Mapping):
        return tuple(moderators.items())
    if isinstance(moderators, str | bytes):  # validated with a domain error later
        return ()
    return tuple((name, name) for name in moderators)


def meta_regression(
    data: pd.DataFrame | None = None,
    *,
    effect: ColumnOrArray,
    variance: ColumnOrArray | None = None,
    standard_error: ColumnOrArray | None = None,
    moderators: ModeratorInput,
    categorical: CategoricalInput | None = None,
    study: ColumnOrArray | None = None,
    model: str = "mixed",
    tau2_method: str = "REML",
    inference_method: str = "normal",
    intercept: bool = True,
    confidence_level: float = 0.95,
    missing: MissingPolicy = "raise",
    atol: float = 1e-10,
    max_iter: int = 1000,
) -> MetaRegressionResult:
    """Fit a common- or mixed-effects meta-regression.

    Moderators may be a sequence of DataFrame column names or a mapping from
    stable moderator names to column names/one-dimensional arrays. Categorical
    moderators must be declared explicitly as ordered level sequences; the
    first level is the treatment-coding reference. Coefficients describe
    study-level associations and do not establish individual-level or causal
    effects.
    """

    confidence_level, atol, max_iter = _validate_analysis_controls(
        confidence_level=confidence_level,
        atol=atol,
        max_iter=max_iter,
    )
    normalized_model = _normalize_regression_model(model)
    normalized_inference = _normalize_inference_method(inference_method)
    normalized_tau2 = tau2_method.upper().replace("-", "_")

    normalized = normalize_meta_regression_data(
        data=data,
        effect=effect,
        variance=variance,
        standard_error=standard_error,
        moderators=moderators,
        categorical=categorical,
        study=study,
        missing=missing,
        intercept=intercept,
    )
    included_effect = normalized.included_effect
    included_variance = normalized.included_variance
    design = normalized.included_design_matrix
    fit = fit_meta_regression(
        included_effect,
        included_variance,
        design,
        intercept=intercept,
        model=normalized_model,
        tau2_method=normalized_tau2,
        inference_method=normalized_inference,
        confidence_level=confidence_level,
        atol=atol,
        max_iter=max_iter,
    )

    residual_df = len(included_effect) - design.shape[1]
    qe, trace_p0 = residual_heterogeneity(included_effect, included_variance, design)
    qe_pvalue = float(chi2.sf(qe, df=residual_df))
    tau2_value = 0.0 if fit.tau2 is None else fit.tau2.value
    if normalized_model == "common":
        residual_i2 = 0.0 if qe <= 0.0 else max(0.0, (qe - residual_df) / qe)
        residual_h2 = qe / residual_df
        i2_method = "q_based_residual"
    else:
        typical_variance = residual_df / trace_p0
        if tau2_value == 0.0:
            residual_i2 = 0.0
            residual_h2 = 1.0
        else:
            residual_i2 = tau2_value / (tau2_value + typical_variance)
            residual_h2 = 1.0 + tau2_value / typical_variance
        i2_method = "tau2_typical_variance_residual"
    heterogeneity = HeterogeneityResult(
        q=qe,
        df=residual_df,
        pvalue=qe_pvalue,
        i2=float(residual_i2),
        h2=float(residual_h2),
        i2_method=i2_method,
    )

    warnings = list(fit.warnings)
    k = len(included_effect)
    p = design.shape[1]
    if k < 10:
        warnings.append(
            "Meta-regression with fewer than 10 studies has limited power and "
            "unstable moderator inference."
        )
    if residual_df <= 4:
        warnings.append(
            f"Only {residual_df} residual degree(s) of freedom remain; coefficient "
            "and heterogeneity inference may be unstable."
        )
    condition_threshold = 1.0 / np.sqrt(np.finfo(np.float64).eps)
    if normalized.condition_number > condition_threshold:
        warnings.append(
            "The moderator design matrix has a high condition number "
            f"({normalized.condition_number:.6g}); coefficient estimates may be "
            "numerically unstable."
        )
    if fit.tau2 is not None and fit.tau2.boundary:
        warnings.append("Residual tau-squared was estimated at the zero boundary.")
    if normalized_model == "mixed" and k < 5:
        warnings.append(
            "Meta-regression prediction intervals are especially uncertain with "
            "fewer than five studies."
        )

    tau2_null: float | None = None
    pseudo_r2: float | None = None
    pseudo_r2_raw: float | None = None
    if normalized_model == "mixed" and intercept:
        null_tau = estimate_meta_regression_tau2(
            included_effect,
            included_variance,
            np.ones((k, 1), dtype=np.float64),
            method=normalized_tau2,
            atol=atol,
            max_iter=max_iter,
        )
        tau2_null = null_tau.value
        if tau2_null == 0.0:
            warnings.append(
                "Pseudo-R-squared is undefined because the intercept-only model "
                "estimated tau-squared at zero."
            )
        else:
            pseudo_r2_raw = 1.0 - tau2_value / tau2_null
            pseudo_r2 = max(0.0, pseudo_r2_raw)
            if pseudo_r2_raw < 0.0:
                warnings.append(
                    "Moderators increased the estimated residual tau-squared; "
                    "pseudo-R-squared was truncated to zero."
                )

    term_to_moderator = {
        term: spec.name
        for spec in normalized.design_info.moderators
        for term in spec.term_names
    }
    coefficient_df = np.nan if fit.coefficient_df is None else float(fit.coefficient_df)
    coefficients = pd.DataFrame(
        {
            "term": normalized.design_info.term_names,
            "moderator": [
                term_to_moderator.get(term)
                for term in normalized.design_info.term_names
            ],
            "estimate": fit.coefficients,
            "standard_error": fit.standard_errors,
            "statistic": fit.statistics,
            "statistic_name": fit.statistic_name,
            "df": coefficient_df,
            "pvalue": fit.pvalues,
            "ci_low": fit.ci_low,
            "ci_high": fit.ci_high,
        }
    )

    row_count = len(normalized.row_id)
    fitted_values = np.full(row_count, np.nan, dtype=np.float64)
    residuals = np.full(row_count, np.nan, dtype=np.float64)
    precision_weights = np.full(row_count, np.nan, dtype=np.float64)
    normalized_weights = np.full(row_count, np.nan, dtype=np.float64)
    leverage = np.full(row_count, np.nan, dtype=np.float64)
    fitted_values[normalized.included] = fit.fitted_values
    residuals[normalized.included] = fit.residuals
    precision_weights[normalized.included] = fit.precision_weights
    normalized_weights[normalized.included] = fit.normalized_precision_weights
    leverage[normalized.included] = fit.leverage
    study_payload: dict[str, object] = {
        "row_id": normalized.row_id,
        "study": normalized.study,
        "effect": normalized.effect,
        "variance": normalized.variance,
        "standard_error": np.sqrt(normalized.variance),
    }
    study_payload.update(
        {name: values.copy() for name, values in normalized.moderator_values}
    )
    study_payload.update(
        {
            "included": normalized.included,
            "exclusion_reason": pd.Series(
                normalized.exclusion_reason, dtype=object, copy=True
            ),
            "fitted_value": fitted_values,
            "residual": residuals,
            "precision_weight": precision_weights,
            "normalized_precision_weight": normalized_weights,
            "leverage": leverage,
        }
    )
    study_results = pd.DataFrame(study_payload)

    excluded_count = int(np.count_nonzero(~normalized.included))
    if excluded_count:
        warnings.append(
            f"Excluded {excluded_count} study row(s) under missing={missing!r}."
        )

    moderator_inputs = _moderator_inputs(moderators)
    uncertainty_input: tuple[str, ColumnOrArray]
    transformations: list[TransformationRecord] = []
    if standard_error is not None:
        uncertainty_input = ("standard_error", standard_error)
        transformations.append(
            TransformationRecord(
                name="standard_error_to_variance",
                affected_rows=tuple(
                    int(row) for row in np.flatnonzero(~np.isnan(normalized.variance))
                ),
            )
        )
    else:
        assert variance is not None  # validated by normalize_studies
        uncertainty_input = ("variance", variance)
    for spec in normalized.design_info.moderators:
        if spec.kind == "categorical":
            transformations.append(
                TransformationRecord(
                    name="categorical_treatment_coding",
                    parameters=(
                        ("moderator", spec.name),
                        ("levels", repr(spec.levels)),
                        ("reference", repr(spec.reference)),
                        ("terms", repr(spec.term_names)),
                    ),
                    affected_rows=tuple(
                        int(row) for row in np.flatnonzero(normalized.included)
                    ),
                )
            )
    provenance = build_analysis_provenance(
        analysis_type="meta_regression",
        data=data,
        inputs=(
            ("effect", effect),
            uncertainty_input,
            *((f"moderator:{name}", value) for name, value in moderator_inputs),
        ),
        study=study,
        included=normalized.included,
        transformations=tuple(transformations),
    )

    method = MetaRegressionMethodConfig(
        model=normalized_model,
        tau2_method=None if normalized_model == "common" else normalized_tau2,
        inference_method=normalized_inference,
        confidence_level=confidence_level,
        intercept=intercept,
        moderator_names=normalized.design_info.moderator_names,
        term_names=normalized.design_info.term_names,
        categorical_references=tuple(
            (spec.name, repr(spec.reference))
            for spec in normalized.design_info.moderators
            if spec.kind == "categorical"
        ),
        prediction_interval_method=(
            "normal_or_t_k_minus_p" if normalized_model == "mixed" else None
        ),
        missing=missing,
        atol=atol,
        max_iter=max_iter,
    )
    global_fit = fit.global_test
    global_test = ModeratorTestResult(
        moderator="all",
        terms=tuple(
            normalized.design_info.term_names[1:]
            if intercept
            else normalized.design_info.term_names
        ),
        statistic=global_fit.statistic,
        statistic_name=global_fit.statistic_name,
        distribution=global_fit.distribution,
        df_num=global_fit.df_num,
        df_denom=global_fit.df_denom,
        pvalue=global_fit.pvalue,
    )
    diagnostics = MetaRegressionDiagnostics(
        converged=True if fit.tau2 is None else fit.tau2.converged,
        iterations=0 if fit.tau2 is None else fit.tau2.iterations,
        tau2_at_boundary=None if fit.tau2 is None else fit.tau2.boundary,
        rank=p,
        condition_number=normalized.condition_number,
        residual_scale=fit.residual_scale,
    )
    return MetaRegressionResult(
        k=k,
        p=p,
        residual_df=residual_df,
        model=normalized_model,
        tau2=tau2_value,
        tau2_null=tau2_null,
        pseudo_r2=pseudo_r2,
        pseudo_r2_raw=pseudo_r2_raw,
        heterogeneity=heterogeneity,
        global_test=global_test,
        method=method,
        diagnostics=diagnostics,
        design_info=normalized.design_info,
        provenance=provenance,
        warnings=tuple(warnings),
        _coefficients=coefficients,
        _coefficient_covariance=fit.covariance,
        _coefficient_vector=fit.coefficients,
        _design_matrix=normalized.design_matrix,
        _study_results=study_results,
        _source_data=data,
    )
