"""Explicit linear contrasts for fitted meta-regression models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, TypeAlias, TypeGuard, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2, f, norm, t

from .exceptions import InvalidStudyDataError

if TYPE_CHECKING:
    from .regression_results import MetaRegressionResult

ContrastWeights: TypeAlias = Mapping[str, float]
NamedContrasts: TypeAlias = Mapping[str, Mapping[str, float]]
ContrastInput: TypeAlias = ContrastWeights | NamedContrasts | pd.DataFrame
ContrastRHS: TypeAlias = float | Mapping[str, float]


@dataclass(frozen=True, slots=True)
class LinearContrastTestResult:
    """Distribution-explicit joint test of one or more linear hypotheses."""

    contrasts: tuple[str, ...]
    statistic: float
    statistic_name: str
    distribution: str
    df_num: int
    df_denom: int | None
    pvalue: float

    def to_dict(self) -> dict[str, object]:
        """Return a detached machine-readable representation."""

        return {
            "contrasts": list(self.contrasts),
            "statistic": self.statistic,
            "statistic_name": self.statistic_name,
            "distribution": self.distribution,
            "df_num": self.df_num,
            "df_denom": self.df_denom,
            "pvalue": self.pvalue,
        }


@dataclass(frozen=True, slots=True)
class MetaRegressionContrastResult:
    """Individual and joint inference for explicit linear contrasts."""

    original: MetaRegressionResult
    joint_test: LinearContrastTestResult
    pvalue_adjustment: str
    warnings: tuple[str, ...]
    _table: pd.DataFrame = field(repr=False, compare=False)
    _contrast_matrix: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_table", self._table.copy(deep=True))
        object.__setattr__(
            self, "_contrast_matrix", self._contrast_matrix.copy(deep=True)
        )

    def __len__(self) -> int:
        return len(self._table)

    @property
    def table(self) -> pd.DataFrame:
        """Return one row of inference per requested contrast."""

        return self._table.copy(deep=True)

    @property
    def contrast_matrix(self) -> pd.DataFrame:
        """Return the labeled coefficient-weight matrix."""

        return self._contrast_matrix.copy(deep=True)

    def summary(self) -> pd.DataFrame:
        """Return the individual contrast table."""

        return self.table

    def to_dataframe(self) -> pd.DataFrame:
        """Return the individual contrast table."""

        return self.table


def _is_real_number(value: object) -> TypeGuard[Real]:
    return isinstance(value, Real) and not isinstance(value, bool)


def _validate_contrast_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidStudyDataError("Contrast names must be non-empty strings.")
    return value


def _weight_vector(
    weights: Mapping[str, object],
    *,
    contrast_name: str,
    term_names: tuple[str, ...],
) -> NDArray[np.float64]:
    if not weights:
        raise InvalidStudyDataError(
            f"Contrast {contrast_name!r} must specify at least one term weight."
        )
    unknown = [term for term in weights if term not in term_names]
    if unknown:
        raise InvalidStudyDataError(
            f"Contrast {contrast_name!r} contains unknown terms: {unknown!r}; "
            f"expected fitted terms {list(term_names)!r}."
        )
    vector = np.zeros(len(term_names), dtype=np.float64)
    for term, weight in weights.items():
        if not _is_real_number(weight):
            raise InvalidStudyDataError(
                f"Weight for term {term!r} in contrast {contrast_name!r} "
                "must be a real number."
            )
        numeric = float(weight)
        if not np.isfinite(numeric):
            raise InvalidStudyDataError(
                f"Weight for term {term!r} in contrast {contrast_name!r} "
                "must be finite."
            )
        vector[term_names.index(term)] = numeric
    if not np.any(vector != 0.0):
        raise InvalidStudyDataError(
            f"Contrast {contrast_name!r} must contain a nonzero term weight."
        )
    return vector


def _from_dataframe(
    contrasts: pd.DataFrame,
    *,
    term_names: tuple[str, ...],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    if contrasts.empty:
        raise InvalidStudyDataError("Contrast DataFrame must contain at least one row.")
    if contrasts.index.has_duplicates:
        raise InvalidStudyDataError("Contrast DataFrame index names must be unique.")
    if contrasts.columns.has_duplicates:
        raise InvalidStudyDataError("Contrast DataFrame term columns must be unique.")
    names = tuple(_validate_contrast_name(value) for value in contrasts.index)
    unknown = [column for column in contrasts.columns if column not in term_names]
    if unknown:
        raise InvalidStudyDataError(
            f"Contrast DataFrame contains unknown terms: {unknown!r}; expected "
            f"fitted terms {list(term_names)!r}."
        )
    rows = [
        _weight_vector(
            cast(Mapping[str, object], contrasts.loc[name].to_dict()),
            contrast_name=name,
            term_names=term_names,
        )
        for name in names
    ]
    return names, np.vstack(rows)


def _normalize_contrasts(
    contrasts: ContrastInput,
    *,
    name: str,
    term_names: tuple[str, ...],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    if isinstance(contrasts, pd.DataFrame):
        if name != "contrast":
            raise InvalidStudyDataError(
                "name= is only supported for a single term-weight mapping."
            )
        return _from_dataframe(contrasts, term_names=term_names)

    if not isinstance(contrasts, Mapping) or not contrasts:
        raise InvalidStudyDataError(
            "contrasts must be a non-empty term-weight mapping, named mapping "
            "of mappings, or DataFrame."
        )
    values = tuple(contrasts.values())
    if all(not isinstance(value, Mapping) for value in values):
        contrast_name = _validate_contrast_name(name)
        vector = _weight_vector(
            cast(ContrastWeights, contrasts),
            contrast_name=contrast_name,
            term_names=term_names,
        )
        return (contrast_name,), vector[np.newaxis, :]
    if all(isinstance(value, Mapping) for value in values):
        if name != "contrast":
            raise InvalidStudyDataError(
                "name= is only supported for a single term-weight mapping."
            )
        named_contrasts = cast(NamedContrasts, contrasts)
        names = tuple(_validate_contrast_name(value) for value in named_contrasts)
        rows = [
            _weight_vector(
                weights,
                contrast_name=contrast_name,
                term_names=term_names,
            )
            for contrast_name, weights in named_contrasts.items()
        ]
        return names, np.vstack(rows)
    raise InvalidStudyDataError(
        "Contrast mappings must contain either only real term weights or only "
        "named term-weight mappings."
    )


def _normalize_rhs(
    rhs: ContrastRHS,
    contrast_names: tuple[str, ...],
) -> NDArray[np.float64]:
    if isinstance(rhs, Mapping):
        missing = [name for name in contrast_names if name not in rhs]
        extra = [name for name in rhs if name not in contrast_names]
        if missing or extra:
            raise InvalidStudyDataError(
                "rhs mapping must contain exactly the contrast names; "
                f"missing={missing!r}, extra={extra!r}."
            )
        values = tuple(rhs[name] for name in contrast_names)
    else:
        values = (rhs,) * len(contrast_names)
    if any(not _is_real_number(value) for value in values):
        raise InvalidStudyDataError("rhs values must be real numbers.")
    resolved = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(resolved)):
        raise InvalidStudyDataError("rhs values must be finite.")
    return resolved


def _validate_full_row_rank(matrix: NDArray[np.float64]) -> None:
    contrast_count = matrix.shape[0]
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != contrast_count:
        raise InvalidStudyDataError(
            "Contrast matrix must have full row rank for a joint test; "
            f"found rank {rank} for {contrast_count} contrasts."
        )


def meta_regression_contrast(
    result: MetaRegressionResult,
    contrasts: ContrastInput,
    *,
    name: str,
    rhs: ContrastRHS,
) -> MetaRegressionContrastResult:
    """Evaluate explicit hypotheses of the form ``C beta = rhs``."""

    term_names = result.design_info.term_names
    contrast_names, matrix = _normalize_contrasts(
        contrasts,
        name=name,
        term_names=term_names,
    )
    _validate_full_row_rank(matrix)
    rhs_values = _normalize_rhs(rhs, contrast_names)
    estimates = result.coefficients["estimate"].to_numpy(dtype=np.float64, copy=True)
    coefficient_covariance = result.coefficient_covariance.to_numpy(
        dtype=np.float64, copy=True
    )
    contrast_estimates = matrix @ estimates
    contrast_covariance = matrix @ coefficient_covariance @ matrix.T
    contrast_covariance = 0.5 * (contrast_covariance + contrast_covariance.T)
    variances = np.diag(contrast_covariance)
    if np.any(variances <= 0.0):  # pragma: no cover - full rank covariance
        raise InvalidStudyDataError(
            "Contrast covariance must have positive diagonal entries."
        )
    standard_errors = np.sqrt(variances)
    differences = contrast_estimates - rhs_values
    statistics = differences / standard_errors
    confidence_level = result.method.confidence_level
    alpha = 1.0 - confidence_level
    if result.method.inference_method == "normal":
        statistic_name = "z"
        distribution = "normal"
        dfs = np.full(len(contrast_names), np.nan)
        pvalues = 2.0 * norm.sf(np.abs(statistics))
        critical = float(norm.ppf(1.0 - alpha / 2.0))
    else:
        statistic_name = "t"
        distribution = "t"
        dfs = np.full(len(contrast_names), float(result.residual_df))
        pvalues = 2.0 * t.sf(np.abs(statistics), df=result.residual_df)
        critical = float(t.ppf(1.0 - alpha / 2.0, df=result.residual_df))
    margin = critical * standard_errors

    try:
        wald = float(differences @ np.linalg.solve(contrast_covariance, differences))
    except np.linalg.LinAlgError as error:  # pragma: no cover - rank checked above
        raise InvalidStudyDataError(
            "Joint contrast covariance could not be solved."
        ) from error
    wald = max(0.0, wald)
    contrast_count = len(contrast_names)
    if result.method.inference_method == "normal":
        joint_test = LinearContrastTestResult(
            contrast_names,
            wald,
            "chi_square",
            "chi_square",
            contrast_count,
            None,
            float(chi2.sf(wald, df=contrast_count)),
        )
    else:
        joint_statistic = wald / contrast_count
        joint_test = LinearContrastTestResult(
            contrast_names,
            joint_statistic,
            "F",
            "F",
            contrast_count,
            result.residual_df,
            float(f.sf(joint_statistic, contrast_count, result.residual_df)),
        )
    table = pd.DataFrame(
        {
            "contrast": contrast_names,
            "estimate": contrast_estimates,
            "rhs": rhs_values,
            "estimate_minus_rhs": differences,
            "standard_error": standard_errors,
            "statistic": statistics,
            "statistic_name": statistic_name,
            "distribution": distribution,
            "df": dfs,
            "pvalue": pvalues,
            "ci_low": contrast_estimates - margin,
            "ci_high": contrast_estimates + margin,
            "confidence_level": confidence_level,
            "pvalue_adjustment": "none",
        }
    )
    contrast_matrix = pd.DataFrame(
        matrix.copy(),
        index=pd.Index(contrast_names, name="contrast"),
        columns=term_names,
    )
    multiple_testing_warning = (
        "Individual contrast p-values are unadjusted for multiple testing; "
        "the joint test evaluates the full prespecified hypothesis set."
    )
    warnings = (multiple_testing_warning,) if contrast_count > 1 else ()
    return MetaRegressionContrastResult(
        original=result,
        joint_test=joint_test,
        pvalue_adjustment="none",
        warnings=warnings,
        _table=table,
        _contrast_matrix=contrast_matrix,
    )
