"""Matplotlib funnel plots built from stable result objects."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from ..results import MetaAnalysisResult
from ._utils import configure_log_axis, default_effect_label, to_display_scale

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def _validate_confidence_level(
    confidence_level: float | None, *, fallback: float
) -> float:
    resolved = fallback if confidence_level is None else confidence_level
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise ValueError("confidence_level must be a number between 0 and 1.")
    numeric = float(resolved)
    if not np.isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    return numeric


def funnel_plot(
    result: MetaAnalysisResult,
    *,
    ax: Axes | None = None,
    effect_label: str | None = None,
    confidence_level: float | None = None,
    show_pseudo_confidence_interval: bool = True,
    warn_on_few_studies: bool = True,
    log_scale: bool | None = None,
) -> Axes:
    """Draw a standard-error funnel plot and return its Matplotlib ``Axes``.

    Pseudo confidence limits are calculated on the model scale around the
    fitted pooled estimate and do not incorporate tau-squared. Ratio measures
    are exponentiated and drawn on a logarithmic x-axis by default. The
    function never calls ``show()``.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - tested without plot extra
        raise ImportError(
            "Funnel plots require Matplotlib; install PyMetaAnalysis[plot]."
        ) from error

    studies = result.study_results
    included = studies.loc[studies["included"]].reset_index(drop=True)
    if included.empty:
        raise ValueError("Funnel plot requires at least one included study.")
    if warn_on_few_studies and len(included) < 10:
        warnings.warn(
            "Funnel plots are difficult to interpret with fewer than 10 studies; "
            "asymmetry indicates possible small-study effects, not necessarily "
            "publication bias.",
            UserWarning,
            stacklevel=2,
        )

    level = _validate_confidence_level(
        confidence_level, fallback=result.method.confidence_level
    )
    effect = included["effect"].to_numpy(dtype=np.float64, copy=True)
    standard_error = np.sqrt(included["variance"].to_numpy(dtype=np.float64, copy=True))
    displayed_effect = to_display_scale(effect, display_scale=result.display_scale)
    displayed_reference = float(
        to_display_scale(
            np.asarray([result.estimate], dtype=np.float64),
            display_scale=result.display_scale,
        )[0]
    )

    use_log_scale = result.display_scale == "exp" if log_scale is None else log_scale
    if use_log_scale and (
        np.any(displayed_effect <= 0.0) or displayed_reference <= 0.0
    ):
        raise ValueError(
            "Funnel plot effects and pooled estimate must be strictly positive "
            "on a logarithmic axis."
        )
    created_axes = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.0))
    if use_log_scale:
        ax.set_xscale("log")
        configure_log_axis(ax)

    maximum_se = float(np.max(standard_error))
    plot_maximum_se = maximum_se * 1.08
    if show_pseudo_confidence_interval:
        critical_value = float(norm.ppf(0.5 + level / 2.0))
        standard_error_grid = np.linspace(0.0, plot_maximum_se, 200)
        lower = to_display_scale(
            result.estimate - critical_value * standard_error_grid,
            display_scale=result.display_scale,
        )
        upper = to_display_scale(
            result.estimate + critical_value * standard_error_grid,
            display_scale=result.display_scale,
        )
        if use_log_scale and (np.any(lower <= 0.0) or np.any(upper <= 0.0)):
            raise ValueError(
                "Funnel plot confidence limits must be strictly positive on a "
                "logarithmic axis."
            )
        ax.fill_betweenx(
            standard_error_grid,
            lower,
            upper,
            color="#dbeafe",
            alpha=0.6,
            zorder=0,
        )
        ax.plot(
            lower,
            standard_error_grid,
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        ax.plot(
            upper,
            standard_error_grid,
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )

    ax.axvline(
        displayed_reference,
        color="#4b5563",
        linestyle="-",
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        displayed_effect,
        standard_error,
        s=42.0,
        marker="o",
        color="#2f6f9f",
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
    )
    ax.set_ylim(plot_maximum_se, 0.0)
    ax.set_xlabel(effect_label or default_effect_label(result))
    ax.set_ylabel("Standard error")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.08)
    if created_axes:
        ax.figure.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.96)
    return ax
