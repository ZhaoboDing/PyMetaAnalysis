"""Matplotlib forest plots built from stable result objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..results import MetaAnalysisResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def _to_display_scale(
    values: NDArray[np.float64], *, display_scale: str
) -> NDArray[np.float64]:
    if display_scale == "identity":
        displayed = values.copy()
    elif display_scale == "exp":
        with np.errstate(over="ignore"):
            displayed = np.exp(values)
    else:
        raise ValueError(f"Unknown display scale {display_scale!r}.")
    if not np.all(np.isfinite(displayed)):
        raise ValueError(
            "Forest plot values are non-finite after display-scale transformation."
        )
    return displayed


def _default_effect_label(result: MetaAnalysisResult) -> str:
    labels = {
        "OR": "Odds ratio",
        "RR": "Risk ratio",
        "RD": "Risk difference",
        "MD": "Mean difference",
        "SMD": "Standardized mean difference",
        "GENERIC": "Effect",
    }
    return labels.get(result.measure, result.measure)


def _default_pooled_label(result: MetaAnalysisResult) -> str:
    return "Common effect" if result.model == "common" else "Random effects"


def _marker_areas(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    largest = float(np.max(weights))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("Forest plot weights must be finite and strictly positive.")
    return 24.0 + 176.0 * weights / largest


def forest_plot(
    result: MetaAnalysisResult,
    *,
    ax: Axes | None = None,
    effect_label: str | None = None,
    pooled_label: str | None = None,
    show_prediction_interval: bool = True,
    show_weights: bool = True,
    null_value: float | None = None,
    log_scale: bool | None = None,
) -> Axes:
    """Draw a forest plot and return its Matplotlib ``Axes``.

    Only included studies are plotted. Study confidence intervals use the
    result confidence level and normal critical value. Ratio measures are
    displayed on their exponentiated scale with a logarithmic axis by default.
    The function never calls ``show()``.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError as error:  # pragma: no cover - tested without plot extra
        raise ImportError(
            "Forest plots require Matplotlib; install PyMetaAnalysis[plot]."
        ) from error

    studies = result.study_results
    included = studies.loc[studies["included"]].reset_index(drop=True)
    if included.empty:
        raise ValueError("Forest plot requires at least one included study.")

    effect = included["effect"].to_numpy(dtype=np.float64, copy=True)
    variance = included["variance"].to_numpy(dtype=np.float64, copy=True)
    weights = included["normalized_weight"].to_numpy(dtype=np.float64, copy=True)
    critical_value = float(norm.ppf(0.5 + result.method.confidence_level / 2.0))
    margin = critical_value * np.sqrt(variance)
    study_low = _to_display_scale(effect - margin, display_scale=result.display_scale)
    study_estimate = _to_display_scale(effect, display_scale=result.display_scale)
    study_high = _to_display_scale(effect + margin, display_scale=result.display_scale)
    pooled_low, pooled_estimate, pooled_high = _to_display_scale(
        np.asarray([result.ci_low, result.estimate, result.ci_high]),
        display_scale=result.display_scale,
    )

    use_log_scale = result.display_scale == "exp" if log_scale is None else log_scale
    resolved_null = (
        (1.0 if result.display_scale == "exp" else 0.0)
        if null_value is None
        else float(null_value)
    )
    if not np.isfinite(resolved_null) or (use_log_scale and resolved_null <= 0.0):
        raise ValueError(
            "null_value must be finite and strictly positive on a logarithmic axis."
        )

    created_axes = ax is None
    if ax is None:
        height = max(4.0, 0.5 * (len(included) + 3))
        _, ax = plt.subplots(figsize=(8.0, height))
    if use_log_scale:
        ax.set_xscale("log")

    y_studies = np.arange(len(included), 0, -1, dtype=np.float64)
    pooled_y = 0.0
    ax.axvline(
        resolved_null,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )
    ax.hlines(
        y_studies,
        study_low,
        study_high,
        colors="#4c4c4c",
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        study_estimate,
        y_studies,
        s=_marker_areas(weights),
        marker="s",
        color="#2f6f9f",
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
    )

    if show_prediction_interval and result.prediction_interval is not None:
        prediction_low, prediction_high = _to_display_scale(
            np.asarray(result.prediction_interval, dtype=np.float64),
            display_scale=result.display_scale,
        )
        ax.hlines(
            pooled_y,
            prediction_low,
            prediction_high,
            colors="#d97706",
            linewidth=2.0,
            zorder=1,
        )
        ax.vlines(
            [prediction_low, prediction_high],
            pooled_y - 0.12,
            pooled_y + 0.12,
            colors="#d97706",
            linewidth=1.5,
            zorder=1,
        )

    diamond = Polygon(
        [
            (pooled_low, pooled_y),
            (pooled_estimate, pooled_y + 0.25),
            (pooled_high, pooled_y),
            (pooled_estimate, pooled_y - 0.25),
        ],
        closed=True,
        facecolor="#1f4e79",
        edgecolor="#1f4e79",
        zorder=3,
    )
    ax.add_patch(diamond)

    labels = [str(label) for label in included["study"]]
    labels.append(pooled_label or _default_pooled_label(result))
    ax.set_yticks(np.concatenate([y_studies, [pooled_y]]), labels=labels)
    ax.set_ylim(-0.65, len(included) + 0.75)
    ax.set_xlabel(effect_label or _default_effect_label(result))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.08)

    if show_weights:
        transform = ax.get_yaxis_transform()
        header_y = len(included) + 0.55
        ax.text(
            1.02,
            header_y,
            "Weight",
            transform=transform,
            ha="left",
            va="center",
            fontweight="bold",
            clip_on=False,
        )
        for y_value, weight in zip(y_studies, weights, strict=True):
            ax.text(
                1.02,
                y_value,
                f"{100.0 * weight:.1f}%",
                transform=transform,
                ha="left",
                va="center",
                clip_on=False,
            )
    if created_axes:
        ax.figure.subplots_adjust(left=0.25, right=0.82, bottom=0.14)
    return ax
