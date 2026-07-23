"""Immutable, inspectable meta-regression results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2, f, norm, t

from .config import MetaRegressionMethodConfig
from .design_matrix import DesignInfo, build_prediction_design_matrix
from .exceptions import InvalidStudyDataError
from .provenance import AnalysisProvenance
from .results import HeterogeneityResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .regression_sensitivity import MetaRegressionLeaveOneOutResult
    from .reporting import ResultReport
else:
    Axes = Any


@dataclass(frozen=True, slots=True)
class ModeratorTestResult:
    """A distribution-explicit Wald test for one or more moderator terms."""

    moderator: str
    terms: tuple[str, ...]
    statistic: float
    statistic_name: str
    distribution: str
    df_num: int
    df_denom: int | None
    pvalue: float

    def to_dict(self) -> dict[str, object]:
        """Return a detached machine-readable representation."""

        return {
            "moderator": self.moderator,
            "terms": list(self.terms),
            "statistic": self.statistic,
            "statistic_name": self.statistic_name,
            "distribution": self.distribution,
            "df_num": self.df_num,
            "df_denom": self.df_denom,
            "pvalue": self.pvalue,
        }


@dataclass(frozen=True, slots=True)
class MetaRegressionDiagnostics:
    """Convergence and design-matrix diagnostics for a fitted regression."""

    converged: bool
    iterations: int
    tau2_at_boundary: bool | None
    rank: int
    condition_number: float
    residual_scale: float


@dataclass(frozen=True, slots=True)
class MetaRegressionSummary:
    """Printable and machine-readable meta-regression summary."""

    _result: MetaRegressionResult = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = self._result
        return {
            "model": result.model,
            "studies": result.k,
            "coefficients": result.coefficients.to_dict(orient="records"),
            "tau2": result.tau2,
            "tau2_method": result.method.tau2_method,
            "tau2_null": result.tau2_null,
            "pseudo_r2": result.pseudo_r2,
            "pseudo_r2_raw": result.pseudo_r2_raw,
            "residual_q": result.heterogeneity.q,
            "residual_q_df": result.heterogeneity.df,
            "residual_q_pvalue": result.heterogeneity.pvalue,
            "residual_i2": result.heterogeneity.i2,
            "residual_h2": result.heterogeneity.h2,
            "global_test": result.global_test.to_dict(),
            "warnings": result.warnings,
        }

    def __str__(self) -> str:
        result = self._result
        level = 100.0 * result.method.confidence_level
        lines = [
            f"Meta-regression ({result.model}-effects)",
            (
                f"Studies: {result.k}; coefficients: {result.p}; "
                f"residual df: {result.residual_df}"
            ),
            "",
            f"Coefficients ({level:g}% CI):",
        ]
        for row in result.coefficients.itertuples(index=False):
            df_suffix = (
                ""
                if result.method.inference_method == "normal"
                else f", df={result.residual_df}"
            )
            lines.append(
                f"- {row.term}: {row.estimate:.6g} "
                f"(SE {row.standard_error:.6g}; {row.ci_low:.6g} to "
                f"{row.ci_high:.6g}; {row.statistic_name}={row.statistic:.6g}"
                f"{df_suffix}; p={row.pvalue:.6g})"
            )
        test = result.global_test
        if test.df_denom is None:
            test_df = str(test.df_num)
        else:
            test_df = f"{test.df_num}, {test.df_denom}"
        lines.extend(
            [
                "",
                (
                    "Global moderator test: "
                    f"{test.statistic_name}({test_df})={test.statistic:.6g}, "
                    f"p={test.pvalue:.6g}"
                ),
                (
                    f"Residual heterogeneity: QE({result.heterogeneity.df})="
                    f"{result.heterogeneity.q:.6g}, "
                    f"p={result.heterogeneity.pvalue:.6g}"
                ),
            ]
        )
        if result.model == "mixed":
            lines.append(f"tau^2: {result.tau2:.6g} ({result.method.tau2_method})")
            if result.pseudo_r2 is not None:
                lines.append(f"Pseudo-R^2: {100.0 * result.pseudo_r2:.2f}%")
        lines.append(f"Inference method: {result.method.inference_method}")
        lines.append(
            "Interpretation note: moderators are study-level associations and "
            "must not be interpreted as individual-level causal effects."
        )
        if result.warnings:
            lines.append("Notes:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MetaRegressionResult:
    """Results of a common- or mixed-effects meta-regression."""

    k: int
    p: int
    residual_df: int
    model: str
    tau2: float
    tau2_null: float | None
    pseudo_r2: float | None
    pseudo_r2_raw: float | None
    heterogeneity: HeterogeneityResult
    global_test: ModeratorTestResult
    method: MetaRegressionMethodConfig
    diagnostics: MetaRegressionDiagnostics
    design_info: DesignInfo
    provenance: AnalysisProvenance
    warnings: tuple[str, ...]
    _coefficients: pd.DataFrame = field(repr=False, compare=False)
    _coefficient_covariance: NDArray[np.float64] = field(repr=False, compare=False)
    _coefficient_vector: NDArray[np.float64] = field(repr=False, compare=False)
    _design_matrix: NDArray[np.float64] = field(repr=False, compare=False)
    _study_results: pd.DataFrame = field(repr=False, compare=False)
    _source_data: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_coefficients", self._coefficients.copy(deep=True))
        object.__setattr__(
            self,
            "_coefficient_covariance",
            self._coefficient_covariance.copy(),
        )
        object.__setattr__(self, "_coefficient_vector", self._coefficient_vector.copy())
        object.__setattr__(self, "_design_matrix", self._design_matrix.copy())
        object.__setattr__(self, "_study_results", self._study_results.copy(deep=True))
        if self._source_data is not None:
            object.__setattr__(self, "_source_data", self._source_data.copy(deep=True))

    @property
    def coefficients(self) -> pd.DataFrame:
        """Return a defensive copy of the coefficient table."""

        return self._coefficients.copy(deep=True)

    @property
    def coefficient_covariance(self) -> pd.DataFrame:
        """Return the labeled coefficient covariance matrix."""

        return pd.DataFrame(
            self._coefficient_covariance.copy(),
            index=self.design_info.term_names,
            columns=self.design_info.term_names,
        )

    @property
    def design_matrix(self) -> pd.DataFrame:
        """Return the included-study design matrix as a defensive copy."""

        included = self._study_results["included"].to_numpy(dtype=bool)
        return pd.DataFrame(
            self._design_matrix[included].copy(),
            columns=self.design_info.term_names,
            index=self._study_results.loc[included, "study"].to_numpy(copy=True),
        )

    @property
    def study_results(self) -> pd.DataFrame:
        """Return all input rows with fitted values and row diagnostics."""

        return self._study_results.copy(deep=True)

    @property
    def excluded_studies(self) -> pd.DataFrame:
        """Return rows excluded by the configured missing-value policy."""

        return self.study_results.loc[lambda frame: ~frame["included"]].reset_index(
            drop=True
        )

    @property
    def source_data(self) -> pd.DataFrame | None:
        """Return a defensive copy of the source DataFrame, when supplied."""

        return None if self._source_data is None else self._source_data.copy(deep=True)

    def summary(self) -> MetaRegressionSummary:
        """Return a printable and machine-readable summary."""

        return MetaRegressionSummary(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the row-level result table."""

        return self.study_results

    def leave_one_out(self) -> MetaRegressionLeaveOneOutResult:
        """Refit the model while omitting each included study once."""

        from .regression_sensitivity import meta_regression_leave_one_out

        return meta_regression_leave_one_out(self)

    def _test_terms(
        self, moderator: str, terms: tuple[str, ...]
    ) -> ModeratorTestResult:
        indices = np.asarray(
            [self.design_info.term_names.index(term) for term in terms], dtype=np.int64
        )
        estimates = self._coefficient_vector[indices]
        covariance = self._coefficient_covariance[np.ix_(indices, indices)]
        try:
            wald = float(estimates @ np.linalg.solve(covariance, estimates))
        except np.linalg.LinAlgError as error:  # pragma: no cover - validated fit
            raise InvalidStudyDataError(
                f"Moderator test for {moderator!r} could not be solved."
            ) from error
        wald = max(0.0, wald)
        term_count = len(terms)
        if self.method.inference_method == "normal":
            return ModeratorTestResult(
                moderator,
                terms,
                wald,
                "chi_square",
                "chi_square",
                term_count,
                None,
                float(chi2.sf(wald, df=term_count)),
            )
        statistic = wald / term_count
        return ModeratorTestResult(
            moderator,
            terms,
            statistic,
            "F",
            "F",
            term_count,
            self.residual_df,
            float(f.sf(statistic, term_count, self.residual_df)),
        )

    def test_moderator(self, moderator: str) -> ModeratorTestResult:
        """Jointly test all encoded terms belonging to one moderator."""

        terms = self.design_info.terms_for(moderator)
        return self._test_terms(moderator, terms)

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Predict mean and, for mixed models, true effects at new moderator values."""

        matrix = build_prediction_design_matrix(new_data, self.design_info)
        estimates = matrix @ self._coefficient_vector
        mean_variance = np.einsum(
            "ij,jk,ik->i", matrix, self._coefficient_covariance, matrix
        )
        mean_se = np.sqrt(np.maximum(0.0, mean_variance))
        alpha = 1.0 - self.method.confidence_level
        if self.method.inference_method == "normal":
            critical = float(norm.ppf(1.0 - alpha / 2.0))
        else:
            critical = float(t.ppf(1.0 - alpha / 2.0, df=self.residual_df))
        margin = critical * mean_se
        payload: dict[str, NDArray[np.float64]] = {
            "estimate": estimates,
            "standard_error": mean_se,
            "ci_low": estimates - margin,
            "ci_high": estimates + margin,
        }
        if self.model == "mixed":
            prediction_se = np.sqrt(np.maximum(0.0, self.tau2 + mean_variance))
            prediction_margin = critical * prediction_se
            payload["pi_low"] = estimates - prediction_margin
            payload["pi_high"] = estimates + prediction_margin
        return pd.DataFrame(payload, index=new_data.index.copy())

    def bubble(
        self,
        *,
        ax: Axes | None = None,
        moderator_label: str | None = None,
        effect_label: str = "Effect",
        show_confidence_interval: bool = True,
        show_prediction_interval: bool = False,
    ) -> Axes:
        """Draw a weighted single-numeric-moderator bubble plot.

        Matplotlib is an optional dependency. Install ``PyMetaAnalysis[plot]``
        before calling this method. Multivariable, categorical, and no-intercept
        fits are rejected because a marginal plotting convention would require
        additional user choices.
        """

        from .plotting import bubble_plot

        return bubble_plot(
            self,
            ax=ax,
            moderator_label=moderator_label,
            effect_label=effect_label,
            show_confidence_interval=show_confidence_interval,
            show_prediction_interval=show_prediction_interval,
        )

    def method_details(self) -> str:
        """Describe the fitted model and its interpretation constraints."""

        model_name = "common-effect" if self.model == "common" else "mixed-effects"
        moderator_text = ", ".join(self.method.moderator_names)
        sentences = [
            f"We fitted a {model_name} meta-regression of {self.k} studies using "
            f"inverse-variance weighting and moderators {moderator_text}."
        ]
        if self.model == "mixed":
            sentences.append(
                f"Residual between-study variance was estimated with "
                f"{self.method.tau2_method} (absolute tolerance "
                f"{self.method.atol:g}; maximum {self.method.max_iter} iterations)."
            )
        sentences.append(
            f"Coefficient inference used {self.method.inference_method} at a "
            f"{100.0 * self.method.confidence_level:g}% confidence level."
        )
        for name, reference in self.method.categorical_references:
            sentences.append(
                f"Categorical moderator {name!r} used treatment coding with "
                f"reference {reference}."
            )
        sentences.append(
            "Moderator coefficients describe study-level associations and do not "
            "establish individual-level or causal effects."
        )
        sentences.append(
            f"Missing inputs were handled with missing={self.method.missing!r}. "
            f"The analysis used PyMetaAnalysis {self.provenance.package_version}."
        )
        return " ".join(sentences)

    def report(self, *, include_studies: bool = True) -> ResultReport:
        """Build a detached structured and Markdown report."""

        from .reporting import build_meta_regression_report

        return build_meta_regression_report(self, include_studies=include_studies)
