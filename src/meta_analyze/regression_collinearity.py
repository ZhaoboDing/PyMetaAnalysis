"""Collinearity diagnostics for fitted meta-regression models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .exceptions import InvalidStudyDataError

if TYPE_CHECKING:
    from .regression_results import MetaRegressionResult


@dataclass(frozen=True, slots=True)
class MetaRegressionCollinearityResult:
    """VIF/GVIF and weighted condition diagnostics for a fitted design."""

    original: MetaRegressionResult
    raw_condition_number: float
    weighted_scaled_condition_number: float
    condition_index_reference: float
    variance_proportion_reference: float
    warnings: tuple[str, ...]
    _term_vif: pd.DataFrame = field(repr=False, compare=False)
    _moderator_gvif: pd.DataFrame = field(repr=False, compare=False)
    _condition_indices: pd.DataFrame = field(repr=False, compare=False)
    _variance_proportions: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_term_vif", self._term_vif.copy(deep=True))
        object.__setattr__(
            self, "_moderator_gvif", self._moderator_gvif.copy(deep=True)
        )
        object.__setattr__(
            self, "_condition_indices", self._condition_indices.copy(deep=True)
        )
        object.__setattr__(
            self,
            "_variance_proportions",
            self._variance_proportions.copy(deep=True),
        )

    @property
    def term_vif(self) -> pd.DataFrame:
        """Return VIF and standard-error inflation for encoded terms."""

        return self._term_vif.copy(deep=True)

    @property
    def moderator_gvif(self) -> pd.DataFrame:
        """Return moderator-level GVIF and dimension-adjusted GSIF."""

        return self._moderator_gvif.copy(deep=True)

    @property
    def condition_indices(self) -> pd.DataFrame:
        """Return singular-value dimensions and their condition indices."""

        return self._condition_indices.copy(deep=True)

    @property
    def variance_proportions(self) -> pd.DataFrame:
        """Return long-form coefficient variance-decomposition proportions."""

        return self._variance_proportions.copy(deep=True)

    @property
    def concerning_dimensions(self) -> pd.DataFrame:
        """Return dimensions meeting both documented collinearity references."""

        return (
            self._condition_indices.loc[lambda frame: frame["concerning"]]
            .reset_index(drop=True)
            .copy(deep=True)
        )


def _classic_covariance(result: MetaRegressionResult) -> NDArray[np.float64]:
    studies = result.study_results
    included = studies["included"].to_numpy(dtype=np.bool_, copy=True)
    variance = studies.loc[included, "variance"].to_numpy(dtype=np.float64, copy=True)
    design = result.design_matrix.to_numpy(dtype=np.float64, copy=True)
    denominator = variance + result.tau2
    variance_scale = float(np.min(denominator))
    relative_weights = variance_scale / denominator
    gram = design.T @ (relative_weights[:, np.newaxis] * design)
    try:
        inverse = np.linalg.solve(gram, np.eye(gram.shape[0]))
    except np.linalg.LinAlgError as error:  # pragma: no cover - fit is full rank
        raise InvalidStudyDataError(
            "Meta-regression coefficient covariance could not be reconstructed."
        ) from error
    inverse = 0.5 * (inverse + inverse.T)
    return variance_scale * inverse


def _covariance_correlation(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0.0):
        raise InvalidStudyDataError(
            "VIF diagnostics require positive coefficient variances."
        )
    standard_errors = np.sqrt(diagonal)
    correlation = covariance / np.outer(standard_errors, standard_errors)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return np.asarray(correlation, dtype=np.float64)


def _log_determinant(
    matrix: NDArray[np.float64], positions: NDArray[np.int64]
) -> float:
    if positions.size == 0:
        return 0.0
    selected = matrix[np.ix_(positions, positions)]
    sign, value = np.linalg.slogdet(selected)
    if sign <= 0.0:
        raise InvalidStudyDataError(
            "VIF diagnostics require a positive-definite coefficient "
            "correlation matrix."
        )
    return float(value)


def _gvif(correlation: NDArray[np.float64], positions: NDArray[np.int64]) -> float:
    all_positions = np.arange(correlation.shape[0], dtype=np.int64)
    complement = all_positions[~np.isin(all_positions, positions)]
    log_gvif = (
        _log_determinant(correlation, positions)
        + _log_determinant(correlation, complement)
        - _log_determinant(correlation, all_positions)
    )
    if log_gvif >= np.log(np.finfo(np.float64).max):
        return float("inf")
    return max(1.0, float(np.exp(log_gvif)))


def _vif_tables(
    result: MetaRegressionResult,
    covariance: NDArray[np.float64],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offset = 1 if result.design_info.intercept else 0
    terms = result.design_info.term_names[offset:]
    correlation = _covariance_correlation(covariance[offset:, offset:])
    moderator_for_term = {
        term: spec.name
        for spec in result.design_info.moderators
        for term in spec.term_names
    }

    term_rows: list[dict[str, object]] = []
    for position, term in enumerate(terms):
        vif = _gvif(correlation, np.asarray([position], dtype=np.int64))
        term_rows.append(
            {
                "term": term,
                "moderator": moderator_for_term[term],
                "vif": vif,
                "sif": float(np.sqrt(vif)),
            }
        )

    moderator_rows: list[dict[str, object]] = []
    for spec in result.design_info.moderators:
        positions = np.asarray(
            [terms.index(term) for term in spec.term_names], dtype=np.int64
        )
        gvif = _gvif(correlation, positions)
        term_count = len(positions)
        moderator_rows.append(
            {
                "moderator": spec.name,
                "kind": spec.kind,
                "terms": spec.term_names,
                "term_count": term_count,
                "gvif": gvif,
                "gsif": float(gvif ** (1.0 / (2.0 * term_count))),
            }
        )
    return pd.DataFrame(term_rows), pd.DataFrame(moderator_rows)


def _condition_tables(
    result: MetaRegressionResult,
    *,
    condition_reference: float,
    variance_reference: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    studies = result.study_results
    included = studies["included"].to_numpy(dtype=np.bool_, copy=True)
    variance = studies.loc[included, "variance"].to_numpy(dtype=np.float64, copy=True)
    design = result.design_matrix.to_numpy(dtype=np.float64, copy=True)
    weights = 1.0 / (variance + result.tau2)
    weighted_design = np.sqrt(weights)[:, np.newaxis] * design
    column_norms = np.linalg.norm(weighted_design, axis=0)
    if np.any(column_norms == 0.0):  # pragma: no cover - full rank checked at fit
        raise InvalidStudyDataError(
            "Condition diagnostics require nonzero weighted design columns."
        )
    scaled_design = weighted_design / column_norms
    _, singular_values, right_vectors_transposed = np.linalg.svd(
        scaled_design, full_matrices=False
    )
    eigenvalues = singular_values * singular_values
    condition_indices = singular_values[0] / singular_values
    variance_components = (
        right_vectors_transposed.T * right_vectors_transposed.T
    ) / eigenvalues[np.newaxis, :]
    variance_proportions = variance_components / np.sum(
        variance_components, axis=1, keepdims=True
    )

    condition_rows: list[dict[str, object]] = []
    variance_rows: list[dict[str, object]] = []
    term_to_moderator = {
        term: spec.name
        for spec in result.design_info.moderators
        for term in spec.term_names
    }
    for position, (singular_value, eigenvalue, condition_index) in enumerate(
        zip(singular_values, eigenvalues, condition_indices, strict=True), start=1
    ):
        high_variance_count = int(
            np.count_nonzero(variance_proportions[:, position - 1] > variance_reference)
        )
        high_condition = bool(condition_index > condition_reference)
        concerning = bool(high_condition and high_variance_count >= 2)
        condition_rows.append(
            {
                "dimension": position,
                "singular_value": singular_value,
                "eigenvalue": eigenvalue,
                "condition_index": condition_index,
                "high_condition_index": high_condition,
                "high_variance_term_count": high_variance_count,
                "concerning": concerning,
            }
        )
        for term, proportion in zip(
            result.design_info.term_names,
            variance_proportions[:, position - 1],
            strict=True,
        ):
            variance_rows.append(
                {
                    "dimension": position,
                    "condition_index": condition_index,
                    "term": term,
                    "moderator": term_to_moderator.get(term),
                    "variance_proportion": proportion,
                    "high_variance_proportion": bool(proportion > variance_reference),
                }
            )
    return (
        pd.DataFrame(condition_rows),
        pd.DataFrame(variance_rows),
        float(condition_indices[-1]),
    )


def meta_regression_collinearity(
    result: MetaRegressionResult,
) -> MetaRegressionCollinearityResult:
    """Compute coefficient inflation and weighted design diagnostics."""

    condition_reference = 30.0
    variance_reference = 0.5
    covariance = _classic_covariance(result)
    term_vif, moderator_gvif = _vif_tables(result, covariance)
    condition_indices, variance_proportions, condition_number = _condition_tables(
        result,
        condition_reference=condition_reference,
        variance_reference=variance_reference,
    )
    warnings: list[str] = []
    if condition_indices["high_condition_index"].any():
        warnings.append(
            "At least one weighted, column-scaled condition index exceeds the "
            "heuristic reference of 30; inspect variance-decomposition "
            "proportions."
        )
    if condition_indices["concerning"].any():
        warnings.append(
            "At least one high-condition dimension concentrates more than 50% "
            "of the coefficient variance for multiple terms."
        )
    return MetaRegressionCollinearityResult(
        original=result,
        raw_condition_number=result.diagnostics.condition_number,
        weighted_scaled_condition_number=condition_number,
        condition_index_reference=condition_reference,
        variance_proportion_reference=variance_reference,
        warnings=tuple(warnings),
        _term_vif=term_vif,
        _moderator_gvif=moderator_gvif,
        _condition_indices=condition_indices,
        _variance_proportions=variance_proportions,
    )
