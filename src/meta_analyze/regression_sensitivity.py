"""Leave-one-out sensitivity workflows for meta-regression."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
