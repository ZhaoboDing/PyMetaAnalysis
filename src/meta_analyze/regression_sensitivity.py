"""Leave-one-out sensitivity workflows for meta-regression."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2, norm

from .data import MissingPolicy
from .exceptions import InsufficientStudiesError, MetaAnalysisError
from .provenance import remap_provenance_rows
from .regression_results import MetaRegressionResult


@dataclass(frozen=True, slots=True)
class MetaRegressionLeaveOneOutResult:
    """Repeated meta-regression fits obtained by omitting each study once."""

    original: MetaRegressionResult
    results: tuple[MetaRegressionResult | None, ...]
    warnings: tuple[str, ...]
    _table: pd.DataFrame = field(repr=False, compare=False)
    _coefficients: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_table", self._table.copy(deep=True))
        object.__setattr__(self, "_coefficients", self._coefficients.copy(deep=True))

    def __len__(self) -> int:
        return len(self.results)

    @property
    def table(self) -> pd.DataFrame:
        """Return one row per omitted study and attempted refit."""

        return self._table.copy(deep=True)

    @property
    def coefficients(self) -> pd.DataFrame:
        """Return coefficient estimates and changes for every deletion."""

        return self._coefficients.copy(deep=True)

    @property
    def failed(self) -> pd.DataFrame:
        """Return deletions for which the reduced model could not be fitted."""

        return self._table.loc[lambda frame: ~frame["refit_success"]].reset_index(
            drop=True
        )

    def summary(self) -> pd.DataFrame:
        """Return the tabular leave-one-out model summary."""

        return self.table

    def to_dataframe(self) -> pd.DataFrame:
        """Return the tabular leave-one-out model summary."""

        return self.table


@dataclass(frozen=True, slots=True)
class MetaRegressionInfluenceResult:
    """Exact deletion diagnostics for a fitted meta-regression."""

    original: MetaRegressionResult
    leave_one_out: MetaRegressionLeaveOneOutResult
    studentized_residual_reference: float
    cook_distance_threshold: float
    dfbetas_threshold: float
    warnings: tuple[str, ...]
    _table: pd.DataFrame = field(repr=False, compare=False)
    _dfbetas: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_table", self._table.copy(deep=True))
        object.__setattr__(self, "_dfbetas", self._dfbetas.copy(deep=True))

    def __len__(self) -> int:
        return len(self.leave_one_out)

    @property
    def results(self) -> tuple[MetaRegressionResult | None, ...]:
        """Return omission-aligned exact refits."""

        return self.leave_one_out.results

    @property
    def table(self) -> pd.DataFrame:
        """Return one row of case-level diagnostics per omitted study."""

        return self._table.copy(deep=True)

    @property
    def dfbetas(self) -> pd.DataFrame:
        """Return long-form coefficient changes standardized by deletion."""

        return self._dfbetas.copy(deep=True)

    @property
    def failed(self) -> pd.DataFrame:
        """Return deletion attempts for which diagnostics are unavailable."""

        return self._table.loc[lambda frame: ~frame["refit_success"]].reset_index(
            drop=True
        )

    @property
    def flagged(self) -> pd.DataFrame:
        """Return rows exceeding at least one documented screening threshold."""

        return self._table.loc[lambda frame: frame["flagged"]].reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """Return the case-level influence table."""

        return self.table

    def to_dataframe(self) -> pd.DataFrame:
        """Return the case-level influence table."""

        return self.table


def _refit_meta_regression(
    result: MetaRegressionResult, positions: NDArray[np.int64]
) -> MetaRegressionResult:
    """Refit a stored meta-regression on selected local row positions."""

    from .regression_api import meta_regression

    studies = result.study_results
    selected = studies.iloc[positions]
    moderator_values = {
        name: selected[name].to_numpy(copy=True)
        for name in result.design_info.moderator_names
    }
    categorical = {
        spec.name: spec.levels
        for spec in result.design_info.moderators
        if spec.kind == "categorical"
    }
    fitted = meta_regression(
        effect=selected["effect"].to_numpy(dtype=np.float64, copy=True),
        variance=selected["variance"].to_numpy(dtype=np.float64, copy=True),
        moderators=moderator_values,
        categorical=categorical or None,
        study=selected["study"].to_numpy(dtype=object, copy=True),
        model=result.model,
        tau2_method=result.method.tau2_method or "REML",
        inference_method=result.method.inference_method,
        intercept=result.method.intercept,
        confidence_level=result.method.confidence_level,
        prediction_interval_method=(
            result.method.prediction_interval_method or "default"
        ),
        missing=cast(MissingPolicy, result.method.missing),
        atol=result.method.atol,
        max_iter=result.method.max_iter,
    )

    row_ids = selected["row_id"].to_numpy(dtype=np.int64, copy=True)
    refitted_studies = fitted.study_results
    refitted_studies["row_id"] = row_ids
    source = result.source_data
    selected_source = None if source is None else source.iloc[positions].copy(deep=True)
    return replace(
        fitted,
        provenance=remap_provenance_rows(fitted.provenance, row_ids.tolist()),
        _study_results=refitted_studies,
        _source_data=selected_source,
    )


def _successful_row(
    source_row: pd.Series[Any], fitted: MetaRegressionResult
) -> dict[str, Any]:
    test = fitted.global_test
    return {
        "omitted_row_id": source_row["row_id"],
        "omitted_study": source_row["study"],
        "refit_success": True,
        "error_type": None,
        "error_message": None,
        "k": fitted.k,
        "tau2": fitted.tau2,
        "residual_q": fitted.heterogeneity.q,
        "residual_q_df": fitted.heterogeneity.df,
        "residual_q_pvalue": fitted.heterogeneity.pvalue,
        "residual_i2": fitted.heterogeneity.i2,
        "residual_h2": fitted.heterogeneity.h2,
        "global_statistic": test.statistic,
        "global_statistic_name": test.statistic_name,
        "global_df_num": test.df_num,
        "global_df_denom": test.df_denom,
        "global_pvalue": test.pvalue,
        "condition_number": fitted.diagnostics.condition_number,
        "refit_warnings": fitted.warnings,
    }


def _failed_row(
    source_row: pd.Series[Any], retained_count: int, error: MetaAnalysisError
) -> dict[str, Any]:
    return {
        "omitted_row_id": source_row["row_id"],
        "omitted_study": source_row["study"],
        "refit_success": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "k": retained_count,
        "tau2": np.nan,
        "residual_q": np.nan,
        "residual_q_df": np.nan,
        "residual_q_pvalue": np.nan,
        "residual_i2": np.nan,
        "residual_h2": np.nan,
        "global_statistic": np.nan,
        "global_statistic_name": None,
        "global_df_num": np.nan,
        "global_df_denom": np.nan,
        "global_pvalue": np.nan,
        "condition_number": np.nan,
        "refit_warnings": (),
    }


def _coefficient_rows(
    *,
    source_row: pd.Series[Any],
    original: pd.DataFrame,
    fitted: MetaRegressionResult | None,
) -> list[dict[str, Any]]:
    if fitted is None:
        deleted = None
    else:
        deleted = {
            str(row["term"]): row
            for row in fitted.coefficients.to_dict(orient="records")
        }

    rows: list[dict[str, Any]] = []
    for original_row in original.to_dict(orient="records"):
        refit_success = deleted is not None
        if deleted is None:
            estimate = np.nan
            standard_error = np.nan
            statistic = np.nan
            statistic_name = None
            coefficient_df = np.nan
            pvalue = np.nan
            ci_low = np.nan
            ci_high = np.nan
        else:
            deleted_row = deleted[str(original_row["term"])]
            estimate = float(deleted_row["estimate"])
            standard_error = float(deleted_row["standard_error"])
            statistic = float(deleted_row["statistic"])
            statistic_name = deleted_row["statistic_name"]
            coefficient_df = float(deleted_row["df"])
            pvalue = float(deleted_row["pvalue"])
            ci_low = float(deleted_row["ci_low"])
            ci_high = float(deleted_row["ci_high"])
        rows.append(
            {
                "omitted_row_id": source_row["row_id"],
                "omitted_study": source_row["study"],
                "refit_success": refit_success,
                "term": original_row["term"],
                "moderator": original_row["moderator"],
                "estimate": estimate,
                "estimate_change": estimate - float(original_row["estimate"]),
                "standard_error": standard_error,
                "statistic": statistic,
                "statistic_name": statistic_name,
                "df": coefficient_df,
                "pvalue": pvalue,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return rows


def meta_regression_leave_one_out(
    result: MetaRegressionResult,
) -> MetaRegressionLeaveOneOutResult:
    """Refit a meta-regression while omitting each included study once."""

    if result.k <= result.p + 1:
        raise InsufficientStudiesError(
            "Meta-regression leave-one-out analysis requires k >= p + 2 so "
            "every reduced model can retain k > p."
        )

    studies = result.study_results
    included = np.flatnonzero(
        studies["included"].to_numpy(dtype=np.bool_, copy=True)
    ).astype(np.int64, copy=False)
    original_coefficients = result.coefficients
    results: list[MetaRegressionResult | None] = []
    rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    failure_count = 0

    for omitted in included:
        retained = included[included != omitted]
        source_row = studies.iloc[int(omitted)]
        try:
            fitted = _refit_meta_regression(result, retained)
        except MetaAnalysisError as error:
            fitted = None
            failure_count += 1
            rows.append(_failed_row(source_row, len(retained), error))
        else:
            assert fitted is not None
            rows.append(_successful_row(source_row, fitted))
        results.append(fitted)
        coefficient_rows.extend(
            _coefficient_rows(
                source_row=source_row,
                original=original_coefficients,
                fitted=fitted,
            )
        )

    warnings: tuple[str, ...] = ()
    if failure_count:
        warnings = (
            f"{failure_count} leave-one-out refit(s) could not be estimated; "
            "inspect failed for details.",
        )
    return MetaRegressionLeaveOneOutResult(
        original=result,
        results=tuple(results),
        warnings=warnings,
        _table=pd.DataFrame(rows),
        _coefficients=pd.DataFrame(coefficient_rows),
    )


def _coefficient_covariance_at_tau2(
    variance: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
    tau2: float,
) -> NDArray[np.float64]:
    """Return the classic covariance using a supplied residual variance."""

    denominator = variance + tau2
    variance_scale = float(np.min(denominator))
    relative_weights = variance_scale / denominator
    gram = design_matrix.T @ (relative_weights[:, np.newaxis] * design_matrix)
    inverse = np.linalg.solve(gram, np.eye(gram.shape[0]))
    inverse = 0.5 * (inverse + inverse.T)
    return variance_scale * inverse


def _standardized_change(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    standardized = np.full_like(numerator, np.nan)
    positive = denominator > 0.0
    standardized[positive] = numerator[positive] / denominator[positive]
    zero_with_change = (denominator == 0.0) & (numerator != 0.0)
    standardized[zero_with_change] = np.sign(numerator[zero_with_change]) * np.inf
    return standardized


def _influence_failed_row(
    source_row: Mapping[Any, Any],
    deletion_row: Mapping[Any, Any],
    *,
    residual_reference: float,
    cook_threshold: float,
    dfbetas_threshold: float,
) -> dict[str, Any]:
    return {
        "omitted_row_id": source_row["row_id"],
        "omitted_study": source_row["study"],
        "refit_success": False,
        "error_type": deletion_row["error_type"],
        "error_message": deletion_row["error_message"],
        "deleted_residual": np.nan,
        "deleted_residual_se": np.nan,
        "externally_standardized_residual": np.nan,
        "studentized_residual_reference": residual_reference,
        "potential_outlier": False,
        "cook_distance": np.nan,
        "cook_distance_threshold": cook_threshold,
        "cook_distance_flag": False,
        "max_abs_dfbetas": np.nan,
        "dfbetas_threshold": dfbetas_threshold,
        "dfbetas_flag": False,
        "potentially_influential": False,
        "flagged": False,
        "leverage": source_row["leverage"],
        "normalized_precision_weight": source_row["normalized_precision_weight"],
    }


def meta_regression_influence(
    result: MetaRegressionResult,
) -> MetaRegressionInfluenceResult:
    """Compute exact deletion residual, Cook's distance, and DFBETAS diagnostics."""

    deletion = meta_regression_leave_one_out(result)
    studies = result.study_results
    included = np.flatnonzero(
        studies["included"].to_numpy(dtype=np.bool_, copy=True)
    ).astype(np.int64, copy=False)
    included_studies = studies.iloc[included].reset_index(drop=True)
    design_matrix = result.design_matrix.to_numpy(dtype=np.float64, copy=True)
    variance = included_studies["variance"].to_numpy(dtype=np.float64, copy=True)
    effect = included_studies["effect"].to_numpy(dtype=np.float64, copy=True)
    full_coefficients = result.coefficients["estimate"].to_numpy(
        dtype=np.float64, copy=True
    )
    full_covariance = result.coefficient_covariance.to_numpy(
        dtype=np.float64, copy=True
    )

    residual_reference = float(norm.ppf(0.975))
    cook_threshold = float(chi2.ppf(0.5, df=result.p))
    dfbetas_threshold = 1.0
    workflow_warnings = list(deletion.warnings)
    try:
        full_precision = np.linalg.solve(
            full_covariance, np.eye(full_covariance.shape[0])
        )
    except np.linalg.LinAlgError:
        full_precision = None
        workflow_warnings.append(
            "Cook's distance is unavailable because the full-model coefficient "
            "covariance is singular."
        )

    rows: list[dict[str, Any]] = []
    dfbetas_rows: list[dict[str, Any]] = []
    original_coefficients = result.coefficients
    for local_position, (source_row, deletion_row, fitted) in enumerate(
        zip(
            included_studies.to_dict(orient="records"),
            deletion.table.to_dict(orient="records"),
            deletion.results,
            strict=True,
        )
    ):
        if fitted is None:
            rows.append(
                _influence_failed_row(
                    source_row,
                    deletion_row,
                    residual_reference=residual_reference,
                    cook_threshold=cook_threshold,
                    dfbetas_threshold=dfbetas_threshold,
                )
            )
            for coefficient in original_coefficients.to_dict(orient="records"):
                dfbetas_rows.append(
                    {
                        "omitted_row_id": source_row["row_id"],
                        "omitted_study": source_row["study"],
                        "refit_success": False,
                        "term": coefficient["term"],
                        "moderator": coefficient["moderator"],
                        "dfbeta": np.nan,
                        "standard_error_reference": np.nan,
                        "dfbetas": np.nan,
                        "threshold": dfbetas_threshold,
                        "exceeds_threshold": False,
                    }
                )
            continue

        deleted_coefficients = fitted.coefficients["estimate"].to_numpy(
            dtype=np.float64, copy=True
        )
        coefficient_change = full_coefficients - deleted_coefficients
        deleted_tau2 = fitted.tau2
        deleted_scale = fitted.diagnostics.residual_scale
        deleted_classic_covariance = _coefficient_covariance_at_tau2(
            variance,
            design_matrix,
            deleted_tau2,
        )
        dfbetas_standard_error = np.sqrt(
            np.maximum(0.0, deleted_scale * np.diag(deleted_classic_covariance))
        )
        dfbetas = _standardized_change(coefficient_change, dfbetas_standard_error)
        finite_abs_dfbetas = np.abs(dfbetas[~np.isnan(dfbetas)])
        max_abs_dfbetas = (
            float(np.max(finite_abs_dfbetas)) if finite_abs_dfbetas.size else np.nan
        )
        dfbetas_flag = bool(np.any(np.abs(dfbetas) > dfbetas_threshold))

        design_row = design_matrix[local_position]
        deleted_prediction = float(design_row @ deleted_coefficients)
        deleted_prediction_variance = float(
            design_row
            @ fitted.coefficient_covariance.to_numpy(dtype=np.float64, copy=True)
            @ design_row
        )
        deleted_residual = float(effect[local_position] - deleted_prediction)
        deleted_residual_variance = deleted_scale * (
            variance[local_position] + deleted_tau2
        ) + max(0.0, deleted_prediction_variance)
        deleted_residual_se = float(np.sqrt(max(0.0, deleted_residual_variance)))
        standardized_residual = _standardized_change(
            np.asarray([deleted_residual], dtype=np.float64),
            np.asarray([deleted_residual_se], dtype=np.float64),
        )[0]
        potential_outlier = bool(abs(standardized_residual) > residual_reference)

        if full_precision is None:
            cook_distance = np.nan
            cook_distance_flag = False
        else:
            cook_distance = max(
                0.0,
                float(coefficient_change @ full_precision @ coefficient_change),
            )
            cook_distance_flag = cook_distance > cook_threshold
        potentially_influential = bool(cook_distance_flag or dfbetas_flag)
        rows.append(
            {
                "omitted_row_id": source_row["row_id"],
                "omitted_study": source_row["study"],
                "refit_success": True,
                "error_type": None,
                "error_message": None,
                "deleted_residual": deleted_residual,
                "deleted_residual_se": deleted_residual_se,
                "externally_standardized_residual": standardized_residual,
                "studentized_residual_reference": residual_reference,
                "potential_outlier": potential_outlier,
                "cook_distance": cook_distance,
                "cook_distance_threshold": cook_threshold,
                "cook_distance_flag": cook_distance_flag,
                "max_abs_dfbetas": max_abs_dfbetas,
                "dfbetas_threshold": dfbetas_threshold,
                "dfbetas_flag": dfbetas_flag,
                "potentially_influential": potentially_influential,
                "flagged": bool(potential_outlier or potentially_influential),
                "leverage": source_row["leverage"],
                "normalized_precision_weight": source_row[
                    "normalized_precision_weight"
                ],
            }
        )
        for coefficient, raw_change, reference_se, standardized in zip(
            original_coefficients.to_dict(orient="records"),
            coefficient_change,
            dfbetas_standard_error,
            dfbetas,
            strict=True,
        ):
            dfbetas_rows.append(
                {
                    "omitted_row_id": source_row["row_id"],
                    "omitted_study": source_row["study"],
                    "refit_success": True,
                    "term": coefficient["term"],
                    "moderator": coefficient["moderator"],
                    "dfbeta": raw_change,
                    "standard_error_reference": reference_se,
                    "dfbetas": standardized,
                    "threshold": dfbetas_threshold,
                    "exceeds_threshold": bool(abs(standardized) > dfbetas_threshold),
                }
            )

    return MetaRegressionInfluenceResult(
        original=result,
        leave_one_out=deletion,
        studentized_residual_reference=residual_reference,
        cook_distance_threshold=cook_threshold,
        dfbetas_threshold=dfbetas_threshold,
        warnings=tuple(workflow_warnings),
        _table=pd.DataFrame(rows),
        _dfbetas=pd.DataFrame(dfbetas_rows),
    )
