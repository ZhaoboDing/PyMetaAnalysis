"""Auditable, JSON-safe reports for fitted meta-analysis results."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .results import MetaAnalysisResult, SubgroupMetaAnalysisResult

REPORT_SCHEMA_VERSION = "1.1"


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime | date | pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(frozen=True, slots=True)
class ResultReport:
    """A detached report with structured, JSON, and Markdown representations."""

    _payload: dict[str, Any] = field(repr=False)
    _markdown: str = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_payload", copy.deepcopy(self._payload))

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive copy of the JSON-compatible report payload."""

        return copy.deepcopy(self._payload)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the report as strict JSON; unavailable values become null."""

        return json.dumps(
            self._payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    def to_markdown(self) -> str:
        """Return a human-readable Markdown report."""

        return self._markdown

    def __str__(self) -> str:
        return self.to_markdown()


def _transformation_rows(result: MetaAnalysisResult, name: str) -> tuple[int, ...]:
    for record in result.provenance.transformations:
        if record.name == name:
            return record.affected_rows
    return ()


def _measure_description(measure: str) -> str:
    return {
        "GENERIC": "generic inverse-variance effects",
        "OR": "log odds ratios",
        "RR": "log risk ratios",
        "RD": "risk differences",
        "MD": "mean differences",
        "SMD": "Hedges' g standardized mean differences",
    }.get(measure, measure)


def method_details(result: MetaAnalysisResult) -> str:
    """Generate a concise, explicit description of the fitted methods."""

    method = result.method
    model_name = "common-effect" if result.model == "common" else "random-effects"
    pooling = (
        "inverse-variance weighting"
        if method.pooling_method == "inverse_variance"
        else "the Mantel–Haenszel estimator"
    )
    study_word = "study" if result.k == 1 else "studies"
    sentences = [
        f"We performed a {model_name} meta-analysis of {result.k} {study_word} "
        f"using {_measure_description(result.measure)}, pooled with {pooling}."
    ]

    if result.model == "random":
        sentences.append(
            "Between-study variance was estimated with "
            f"{method.tau2_method} (absolute tolerance {method.atol:g}; "
            f"maximum {method.max_iter} iterations)."
        )

    ci_description = {
        "normal": "a two-sided normal approximation",
        "hartung_knapp": "a two-sided Hartung–Knapp t interval",
        "hartung_knapp_adhoc": (
            "a two-sided Hartung–Knapp t interval with the ad hoc "
            "lower-bound variance safeguard"
        ),
    }.get(method.ci_method, method.ci_method)
    sentences.append(
        f"The {100.0 * method.confidence_level:g}% confidence interval used "
        f"{ci_description}."
    )
    if result.i2_method == "tau2_typical_variance":
        sentences.append(
            "Cochran's Q used common-effect inverse-variance weights; I² and H² "
            "used tau-squared and the typical within-study variance, where the "
            "latter was derived from those common-effect weights."
        )
    else:
        sentences.append(
            "Cochran's Q, I², and H² used the classical Q-based definition."
        )

    if method.prediction_interval_method is not None:
        if result.prediction_interval is not None:
            sentences.append(
                "A Higgins–Thompson–Spiegelhalter prediction interval was calculated."
            )
        else:
            sentences.append(
                "The configured Higgins–Thompson–Spiegelhalter prediction "
                "interval was unavailable because fewer than three studies "
                "were included."
            )

    options = dict(method.options)
    if result.measure in {"OR", "RR", "RD"}:
        correction = float(options.get("continuity_correction", 0.0) or 0.0)
        scope = str(options.get("correction_scope", "none"))
        affected = _transformation_rows(result, "continuity_correction")
        sentences.append(
            "Individual-study effects used continuity correction "
            f"{correction:g} with scope {scope!r}; it affected "
            f"{len(affected)} row(s)."
        )
        if result.measure == "RD":
            rd_policy = str(options.get("rd_zero_variance", "correct"))
            rd_affected = _transformation_rows(result, "rd_zero_variance_policy")
            if rd_policy == "correct":
                sentences.append(
                    "Risk-difference studies with zero uncorrected sampling "
                    "variance were retained with their raw effect; corrected "
                    "counts were used only for sampling variance. This affected "
                    f"{len(rd_affected)} row(s)."
                )
            else:
                sentences.append(
                    "Risk-difference studies with zero uncorrected sampling "
                    "variance were excluded before pooling and heterogeneity "
                    f"calculations. This affected {len(rd_affected)} row(s)."
                )
        if method.pooling_method == "mantel_haenszel":
            mh_correction = float(options.get("mh_continuity_correction", 0.0) or 0.0)
            mh_scope = str(options.get("mh_correction_scope", "none"))
            mh_affected = _transformation_rows(
                result, "mantel_haenszel_continuity_correction"
            )
            sentences.append(
                "Mantel–Haenszel pooling used continuity correction "
                f"{mh_correction:g} with scope {mh_scope!r}; it affected "
                f"{len(mh_affected)} row(s)."
            )
    elif result.measure == "SMD":
        sentences.append(
            "Standardized mean differences used a pooled standard deviation, "
            "the exact Hedges small-sample correction, and the LS sampling "
            "variance convention."
        )
    elif result.measure == "MD":
        sentences.append("Mean differences used unpooled sampling variances.")

    sentences.append(
        f"Missing inputs were handled with missing={method.missing!r}. "
        f"The analysis used PyMetaAnalysis {result.provenance.package_version}."
    )
    return " ".join(sentences)


def subgroup_method_details(result: SubgroupMetaAnalysisResult) -> str:
    """Describe the overall fit and the subgroup-comparison assumptions."""

    base = method_details(result.overall)
    strategy = (
        "between-study variance was estimated independently within each subgroup"
        if result.method.tau2_strategy == "independent"
        else "between-study variance was not applicable to the common-effect fits"
    )
    return (
        f"{base} Subgroup effects were fitted separately; {strategy}. "
        "Subgroup differences were tested with a fixed-effect inverse-variance "
        "Wald Q test on the subgroup summary estimates."
    )


def _method_payload(result: MetaAnalysisResult) -> dict[str, Any]:
    method = result.method
    return {
        "model": method.model,
        "pooling_method": method.pooling_method,
        "tau2_method": method.tau2_method,
        "ci_method": method.ci_method,
        "confidence_level": method.confidence_level,
        "prediction_interval_method": method.prediction_interval_method,
        "missing": method.missing,
        "atol": method.atol,
        "max_iter": method.max_iter,
        "options": dict(method.options),
    }


def _meta_payload(
    result: MetaAnalysisResult,
    *,
    include_studies: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "analysis": {
            "model": result.model,
            "measure": result.measure,
            "effect_scale": result.effect_scale,
            "display_scale": result.display_scale,
            "included_studies": result.k,
            "total_rows": result.provenance.row_count,
        },
        "results": {
            "estimate": result.estimate,
            "standard_error": result.standard_error,
            "ci": list(result.ci),
            "prediction_interval": result.prediction_interval,
            "display_estimate": result.display_estimate,
            "display_ci": list(result.display_ci),
            "display_prediction_interval": result.display_prediction_interval,
            "tau2": result.tau2,
        },
        "heterogeneity": {
            "q": result.q,
            "df": result.q_df,
            "pvalue": result.q_pvalue,
            "i2": result.i2,
            "h2": result.h2,
            "i2_method": result.i2_method,
        },
        "method": _method_payload(result),
        "diagnostics": {
            "converged": result.diagnostics.converged,
            "iterations": result.diagnostics.iterations,
            "tau2_at_boundary": result.diagnostics.tau2_at_boundary,
        },
        "provenance": result.provenance.to_dict(),
        "warnings": list(result.warnings),
    }
    if include_studies:
        payload["studies"] = result.study_results.to_dict(orient="records")
    return cast(dict[str, Any], _json_safe(payload))


def _number(value: Any) -> str:
    if value is None:
        return "not available"
    numeric = float(value)
    return f"{numeric:.6g}" if math.isfinite(numeric) else "not available"


def _meta_markdown(result: MetaAnalysisResult) -> str:
    display_ci = result.display_ci
    lines = [
        "# Meta-analysis report",
        "",
        "## Results",
        "",
        f"- Model: {result.model}-effect {result.measure}",
        (f"- Studies: {result.k} included of {result.provenance.row_count} input rows"),
        (
            f"- Estimate: {_number(result.display_estimate)} "
            f"({100.0 * result.method.confidence_level:g}% CI "
            f"{_number(display_ci[0])} to {_number(display_ci[1])})"
        ),
        f"- Heterogeneity: Q({result.q_df})={_number(result.q)}, "
        f"p={_number(result.q_pvalue)}; I²={_number(100.0 * result.i2)}%; "
        f"H²={_number(result.h2)} ({result.i2_method})",
    ]
    if result.model == "random":
        lines.append(f"- τ²: {_number(result.tau2)} ({result.method.tau2_method})")
    if result.display_scale != result.effect_scale:
        lines.append(
            f"- Model-scale estimate: {_number(result.estimate)} "
            f"({result.effect_scale} scale)"
        )
    if result.display_prediction_interval is not None:
        low, high = result.display_prediction_interval
        lines.append(f"- Prediction interval: {_number(low)} to {_number(high)}")

    lines.extend(
        [
            "",
            "## Methods",
            "",
            method_details(result),
            "",
            "## Provenance and diagnostics",
            "",
            f"- Package version: {result.provenance.package_version}",
            f"- Provenance schema: {result.provenance.schema_version}",
            f"- Input source: {result.provenance.data_source}",
            f"- Converged: {result.diagnostics.converged}",
            f"- Iterations: {result.diagnostics.iterations}",
            (f"- Excluded row IDs: {list(result.provenance.excluded_rows) or 'none'}"),
        ]
    )
    if result.warnings:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def build_report(
    result: MetaAnalysisResult,
    *,
    include_studies: bool = True,
) -> ResultReport:
    """Build a detached report for a single meta-analysis result."""

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "meta_analysis",
        **_meta_payload(result, include_studies=include_studies),
        "method_details": method_details(result),
    }
    return ResultReport(_json_safe(payload), _meta_markdown(result))


def _subgroup_markdown(result: SubgroupMetaAnalysisResult) -> str:
    lines = [
        "# Subgroup meta-analysis report",
        "",
        "## Results",
        "",
        "| Subgroup | Studies | Estimate | Confidence interval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, group in result.groups.items():
        low, high = group.display_ci
        safe_label = str(label).replace("|", "\\|")
        lines.append(
            f"| {safe_label} | {group.k} | {_number(group.display_estimate)} | "
            f"{_number(low)} to {_number(high)} |"
        )
    lines.extend(
        [
            "",
            (
                "Test for subgroup differences: "
                f"Q({result.q_between_df})={_number(result.q_between)}, "
                f"p={_number(result.q_between_pvalue)}, "
                f"I²={_number(100.0 * result.i2_between)}%."
            ),
            "",
            "## Methods",
            "",
            subgroup_method_details(result),
            "",
            "## Provenance and diagnostics",
            "",
            f"- Package version: {result.overall.provenance.package_version}",
            f"- Input source: {result.overall.provenance.data_source}",
            f"- τ² strategy: {result.method.tau2_strategy}",
            f"- Subgroup test: {result.method.test_method}",
        ]
    )
    warnings = (*result.overall.warnings, *result.warnings)
    if warnings:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def build_subgroup_report(
    result: SubgroupMetaAnalysisResult,
    *,
    include_studies: bool = True,
) -> ResultReport:
    """Build a detached report for a subgroup meta-analysis result."""

    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "subgroup_meta_analysis",
        "overall": _meta_payload(result.overall, include_studies=False),
        "groups": [
            {
                "label": _json_safe(label),
                "result": _meta_payload(group, include_studies=False),
            }
            for label, group in result.groups.items()
        ],
        "subgroup_differences": {
            "q": result.q_between,
            "df": result.q_between_df,
            "pvalue": result.q_between_pvalue,
            "i2": result.i2_between,
            "tau2_strategy": result.method.tau2_strategy,
            "test_method": result.method.test_method,
            "subgroup_missing": result.method.subgroup_missing,
        },
        "method_details": subgroup_method_details(result),
        "warnings": list(result.warnings),
    }
    if include_studies:
        payload["studies"] = result.study_results.to_dict(orient="records")
    return ResultReport(_json_safe(payload), _subgroup_markdown(result))
