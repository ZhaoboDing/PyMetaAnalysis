"""Forest plots for independent study subgroup analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..results import MetaAnalysisResult, SubgroupMetaAnalysisResult
from ._utils import (
    configure_log_axis,
    default_effect_label,
    marker_areas,
    to_display_scale,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def _draw_interval(
    ax: Axes,
    result: MetaAnalysisResult,
    *,
    y: float,
    display_scale: str,
) -> None:
    if result.prediction_interval is None:
        return
    low, high = to_display_scale(
        np.asarray(result.prediction_interval, dtype=np.float64),
        display_scale=display_scale,
    )
    ax.hlines(y, low, high, colors="#d97706", linewidth=2.0, zorder=1)
    ax.vlines(
        [low, high],
        y - 0.11,
        y + 0.11,
        colors="#d97706",
        linewidth=1.4,
        zorder=1,
    )


def _draw_diamond(
    ax: Axes,
    result: MetaAnalysisResult,
    *,
    y: float,
    color: str,
    display_scale: str,
) -> None:
    from matplotlib.patches import Polygon

    low, estimate, high = to_display_scale(
        np.asarray([result.ci_low, result.estimate, result.ci_high]),
        display_scale=display_scale,
    )
    ax.add_patch(
        Polygon(
            [
                (low, y),
                (estimate, y + 0.23),
                (high, y),
                (estimate, y - 0.23),
            ],
            closed=True,
            facecolor=color,
            edgecolor=color,
            zorder=3,
        )
    )


def _study_coordinates(
    result: MetaAnalysisResult,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    studies = result.study_results
    included = studies.loc[studies["included"]]
    effect = included["effect"].to_numpy(dtype=np.float64, copy=True)
    variance = included["variance"].to_numpy(dtype=np.float64, copy=True)
    critical_value = float(norm.ppf(0.5 + result.method.confidence_level / 2.0))
    margin = critical_value * np.sqrt(variance)
    return (
        to_display_scale(effect - margin, display_scale=result.display_scale),
        to_display_scale(effect, display_scale=result.display_scale),
        to_display_scale(effect + margin, display_scale=result.display_scale),
    )


def subgroup_forest_plot(
    result: SubgroupMetaAnalysisResult,
    *,
    ax: Axes | None = None,
    effect_label: str | None = None,
    show_prediction_interval: bool = True,
    show_weights: bool = True,
    null_value: float | None = None,
    log_scale: bool | None = None,
) -> Axes:
    """Draw studies, subgroup summaries, and the overall pooled result."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - tested without plot extra
        raise ImportError(
            "Subgroup forest plots require Matplotlib; install PyMetaAnalysis[plot]."
        ) from error

    overall = result.overall
    use_log_scale = overall.display_scale == "exp" if log_scale is None else log_scale
    resolved_null = (
        (1.0 if overall.display_scale == "exp" else 0.0)
        if null_value is None
        else float(null_value)
    )
    if not np.isfinite(resolved_null) or (use_log_scale and resolved_null <= 0.0):
        raise ValueError(
            "null_value must be finite and strictly positive on a logarithmic axis."
        )

    included_count = sum(group.k for group in result.groups.values())
    row_units = included_count + 2 * len(result.groups) + 1
    created_axes = ax is None
    if ax is None:
        height = max(5.0, 0.42 * row_units + 1.5)
        _, ax = plt.subplots(figsize=(9.0, height))
    if use_log_scale:
        ax.set_xscale("log")
        configure_log_axis(ax)

    overall_studies = overall.study_results.set_index("row_id")
    cursor = float(row_units - 1)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    tick_kinds: list[str] = []
    study_weight_rows: list[tuple[float, float]] = []

    ax.axvline(
        resolved_null,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )

    for label, group in result.groups.items():
        header_y = cursor
        tick_positions.append(header_y)
        tick_labels.append(str(label))
        tick_kinds.append("header")
        cursor -= 1.0

        studies = group.study_results
        included = studies.loc[studies["included"]].reset_index(drop=True)
        low, estimate, high = _study_coordinates(group)
        y_studies = np.arange(
            cursor,
            cursor - len(included),
            -1.0,
            dtype=np.float64,
        )
        row_ids = included["row_id"].to_numpy(dtype=np.int64, copy=True)
        weights = overall_studies.loc[row_ids, "normalized_weight"].to_numpy(
            dtype=np.float64, copy=True
        )
        ax.hlines(
            y_studies,
            low,
            high,
            colors="#4c4c4c",
            linewidth=1.2,
            zorder=1,
        )
        ax.scatter(
            estimate,
            y_studies,
            s=marker_areas(weights),
            marker="s",
            color="#2f6f9f",
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
        )
        for y_value, study, weight in zip(
            y_studies,
            included["study"],
            weights,
            strict=True,
        ):
            tick_positions.append(float(y_value))
            tick_labels.append(f"  {study}")
            tick_kinds.append("study")
            study_weight_rows.append((float(y_value), float(weight)))
        cursor -= float(len(included))

        pooled_y = cursor
        if show_prediction_interval:
            _draw_interval(
                ax,
                group,
                y=pooled_y,
                display_scale=overall.display_scale,
            )
        _draw_diamond(
            ax,
            group,
            y=pooled_y,
            color="#4f7f5f",
            display_scale=overall.display_scale,
        )
        tick_positions.append(pooled_y)
        tick_labels.append(f"  {label} subtotal")
        tick_kinds.append("subtotal")
        cursor -= 1.0

    overall_y = cursor
    if show_prediction_interval:
        _draw_interval(
            ax,
            overall,
            y=overall_y,
            display_scale=overall.display_scale,
        )
    _draw_diamond(
        ax,
        overall,
        y=overall_y,
        color="#1f4e79",
        display_scale=overall.display_scale,
    )
    tick_positions.append(overall_y)
    tick_labels.append("Overall")
    tick_kinds.append("overall")

    ax.set_yticks(tick_positions, labels=tick_labels)
    for tick, kind in zip(ax.get_yticklabels(), tick_kinds, strict=True):
        if kind in {"header", "subtotal", "overall"}:
            tick.set_fontweight("bold")
    ax.set_ylim(overall_y - 0.75, row_units - 0.25)
    ax.set_xlabel(effect_label or default_effect_label(overall))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.08)

    if show_weights:
        transform = ax.get_yaxis_transform()
        ax.text(
            1.02,
            row_units - 0.5,
            "Weight",
            transform=transform,
            ha="left",
            va="center",
            fontweight="bold",
            clip_on=False,
        )
        for weight_y, weight in study_weight_rows:
            ax.text(
                1.02,
                weight_y,
                f"{100.0 * weight:.1f}%",
                transform=transform,
                ha="left",
                va="center",
                clip_on=False,
            )

    test_text = (
        "Test for subgroup differences: "
        f"Q({result.q_between_df})={result.q_between:.3g}, "
        f"p={result.q_between_pvalue:.3g}"
    )
    ax.text(0.0, -0.09, test_text, transform=ax.transAxes, ha="left", va="top")
    if created_axes:
        ax.figure.subplots_adjust(left=0.29, right=0.82, bottom=0.17)
    return ax
