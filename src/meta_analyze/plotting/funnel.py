"""Matplotlib funnel plots built from stable result objects."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
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


def _validate_contour_levels(
    contour_levels: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if contour_levels is None:
        return None
    if isinstance(contour_levels, (str, bytes)):
        raise ValueError(
            "contour_levels must be a strictly increasing sequence between 0 and 1."
        )
    try:
        levels = tuple(contour_levels)
    except TypeError as error:
        raise ValueError(
            "contour_levels must be a strictly increasing sequence between 0 and 1."
        ) from error
    if not levels:
        raise ValueError(
            "contour_levels cannot be empty; use None to disable contours."
        )
    if any(
        isinstance(level, bool)
        or not isinstance(level, Real)
        or not np.isfinite(level)
        or not 0.0 < float(level) < 1.0
        for level in levels
    ):
        raise ValueError("every contour level must be a number between 0 and 1.")
    numeric = tuple(float(level) for level in levels)
    if any(
        next_level <= level
        for level, next_level in zip(numeric[:-1], numeric[1:], strict=True)
    ):
        raise ValueError(
            "contour_levels must be strictly increasing without duplicates."
        )
    return numeric


def _contour_reference_on_model_scale(
    reference: float | None,
    *,
    display_scale: str,
    use_log_scale: bool,
) -> tuple[float, float]:
    default = 1.0 if display_scale == "exp" else 0.0
    resolved = default if reference is None else reference
    if (
        isinstance(resolved, bool)
        or not isinstance(resolved, Real)
        or not np.isfinite(resolved)
    ):
        raise ValueError("contour_reference must be finite on the display scale.")
    displayed = float(resolved)
    if use_log_scale and displayed <= 0.0:
        raise ValueError(
            "contour_reference must be strictly positive on a logarithmic axis."
        )
    if display_scale == "identity":
        model = displayed
    elif display_scale == "exp":
        if displayed <= 0.0:
            raise ValueError(
                "contour_reference must be strictly positive for ratio measures."
            )
        model = math.log(displayed)
    elif display_scale == "tanh":
        if not -1.0 < displayed < 1.0:
            raise ValueError(
                "contour_reference must be strictly between -1 and 1 for correlations."
            )
        model = float(np.arctanh(displayed))
    else:
        raise ValueError(f"Unknown display scale {display_scale!r}.")
    return displayed, model


def _contour_labels(levels: tuple[float, ...]) -> tuple[str, ...]:
    thresholds = tuple(1.0 - level for level in levels)
    labels: list[str] = []
    for index, threshold in enumerate(thresholds):
        if index + 1 < len(thresholds):
            labels.append(f"{thresholds[index + 1]:.3g} < p <= {threshold:.3g}")
        else:
            labels.append(f"p <= {threshold:.3g}")
    return tuple(labels)


def funnel_plot(
    result: MetaAnalysisResult,
    *,
    ax: Axes | None = None,
    effect_label: str | None = None,
    confidence_level: float | None = None,
    show_pseudo_confidence_interval: bool = True,
    contour_levels: Sequence[float] | None = None,
    contour_colors: Sequence[str] | None = None,
    contour_reference: float | None = None,
    show_contour_legend: bool = True,
    warn_on_few_studies: bool = True,
    log_scale: bool | None = None,
) -> Axes:
    """Draw a standard-error funnel plot and return its Matplotlib ``Axes``.

    Pseudo confidence limits are calculated on the model scale around the
    fitted pooled estimate and do not incorporate tau-squared. Ratio measures
    are exponentiated and drawn on a logarithmic x-axis by default. The
    optional contours show two-sided significance bands around a display-scale
    null reference. The function never calls ``show()``.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import is_color_like
        from matplotlib.patches import Patch
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
    resolved_contour_levels = _validate_contour_levels(contour_levels)
    if resolved_contour_levels is None and (
        contour_colors is not None or contour_reference is not None
    ):
        raise ValueError("contour_colors and contour_reference require contour_levels.")
    if not isinstance(show_contour_legend, bool):
        raise ValueError("show_contour_legend must be a boolean.")
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
    maximum_se = float(np.max(standard_error))
    plot_maximum_se = maximum_se * 1.08
    standard_error_grid = np.linspace(0.0, plot_maximum_se, 200)
    pseudo_lower: NDArray[np.float64] | None = None
    pseudo_upper: NDArray[np.float64] | None = None
    if show_pseudo_confidence_interval:
        critical_value = float(norm.ppf(0.5 + level / 2.0))
        pseudo_lower = to_display_scale(
            result.estimate - critical_value * standard_error_grid,
            display_scale=result.display_scale,
        )
        pseudo_upper = to_display_scale(
            result.estimate + critical_value * standard_error_grid,
            display_scale=result.display_scale,
        )
        if use_log_scale and (
            np.any(pseudo_lower <= 0.0) or np.any(pseudo_upper <= 0.0)
        ):
            raise ValueError(
                "Funnel plot confidence limits must be strictly positive on a "
                "logarithmic axis."
            )

    contour_boundaries: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    contour_palette: tuple[Any, ...] = ()
    contour_xlim: tuple[float, float] | None = None
    displayed_contour_reference: float | None = None
    if resolved_contour_levels is not None:
        displayed_contour_reference, contour_reference_model = (
            _contour_reference_on_model_scale(
                contour_reference,
                display_scale=result.display_scale,
                use_log_scale=use_log_scale,
            )
        )
        if contour_colors is None:
            grays = (
                np.asarray([0.70])
                if len(resolved_contour_levels) == 1
                else np.linspace(0.90, 0.50, len(resolved_contour_levels))
            )
            contour_palette = tuple((gray, gray, gray, 1.0) for gray in grays)
        else:
            if isinstance(contour_colors, (str, bytes)):
                raise ValueError(
                    "contour_colors must contain one color per contour level."
                )
            try:
                contour_palette = tuple(contour_colors)
            except TypeError as error:
                raise ValueError(
                    "contour_colors must contain one color per contour level."
                ) from error
            if len(contour_palette) != len(resolved_contour_levels):
                raise ValueError(
                    "contour_colors must contain one color per contour level."
                )
            if any(
                not isinstance(color, str) or not is_color_like(color)
                for color in contour_palette
            ):
                raise ValueError(
                    "every contour color must be a valid Matplotlib color."
                )

        for contour_level in resolved_contour_levels:
            contour_critical = float(norm.ppf(0.5 + contour_level / 2.0))
            contour_boundaries.append(
                (
                    to_display_scale(
                        contour_reference_model
                        - contour_critical * standard_error_grid,
                        display_scale=result.display_scale,
                    ),
                    to_display_scale(
                        contour_reference_model
                        + contour_critical * standard_error_grid,
                        display_scale=result.display_scale,
                    ),
                )
            )
        if use_log_scale and any(
            np.any(lower <= 0.0) or np.any(upper <= 0.0)
            for lower, upper in contour_boundaries
        ):
            raise ValueError(
                "Funnel plot contour limits must be strictly positive on a "
                "logarithmic axis."
            )

        model_coordinates = [effect, np.asarray([result.estimate])]
        if show_pseudo_confidence_interval:
            model_coordinates.extend(
                [
                    result.estimate - critical_value * standard_error_grid,
                    result.estimate + critical_value * standard_error_grid,
                ]
            )
        outer_critical = float(norm.ppf(0.5 + resolved_contour_levels[-1] / 2.0))
        model_coordinates.extend(
            [
                contour_reference_model - outer_critical * standard_error_grid,
                contour_reference_model + outer_critical * standard_error_grid,
            ]
        )
        model_minimum = min(float(np.min(values)) for values in model_coordinates)
        model_maximum = max(float(np.max(values)) for values in model_coordinates)
        if use_log_scale:
            displayed_endpoints = to_display_scale(
                np.asarray([model_minimum, model_maximum], dtype=np.float64),
                display_scale=result.display_scale,
            )
            if np.any(displayed_endpoints <= 0.0):
                raise ValueError(
                    "Funnel plot contour range must be strictly positive on a "
                    "logarithmic axis."
                )
            log_endpoints = np.log(displayed_endpoints)
            log_span = float(log_endpoints[1] - log_endpoints[0])
            log_margin = 0.08 * (log_span if log_span > 0.0 else 1.0)
            with np.errstate(over="ignore", under="ignore"):
                contour_xlim_array = np.exp(
                    np.asarray(
                        [
                            log_endpoints[0] - log_margin,
                            log_endpoints[1] + log_margin,
                        ]
                    )
                )
        else:
            model_span = model_maximum - model_minimum
            model_margin = 0.08 * (model_span if model_span > 0.0 else 1.0)
            contour_xlim_array = to_display_scale(
                np.asarray(
                    [model_minimum - model_margin, model_maximum + model_margin],
                    dtype=np.float64,
                ),
                display_scale=result.display_scale,
            )
        if np.any(~np.isfinite(contour_xlim_array)):
            raise ValueError("Funnel plot contour range is non-finite.")
        contour_xlim = float(contour_xlim_array[0]), float(contour_xlim_array[1])

    created_axes = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.0))
    if use_log_scale:
        ax.set_xscale("log")
        configure_log_axis(ax)

    if contour_xlim is not None:
        for (contour_lower, contour_upper), color in zip(
            contour_boundaries, contour_palette, strict=True
        ):
            ax.fill_betweenx(
                standard_error_grid,
                contour_xlim[0],
                contour_lower,
                color=color,
                alpha=0.85,
                zorder=0,
            )
            ax.fill_betweenx(
                standard_error_grid,
                contour_upper,
                contour_xlim[1],
                color=color,
                alpha=0.85,
                zorder=0,
            )

    if show_pseudo_confidence_interval:
        assert pseudo_lower is not None
        assert pseudo_upper is not None
        if resolved_contour_levels is None:
            ax.fill_betweenx(
                standard_error_grid,
                pseudo_lower,
                pseudo_upper,
                color="#dbeafe",
                alpha=0.6,
                zorder=0,
            )
        ax.plot(
            pseudo_lower,
            standard_error_grid,
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        ax.plot(
            pseudo_upper,
            standard_error_grid,
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )

    if displayed_contour_reference is not None:
        ax.axvline(
            displayed_contour_reference,
            color="#111827",
            linestyle=":",
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
    if contour_xlim is not None:
        ax.set_xlim(*contour_xlim)
    if resolved_contour_levels is not None and show_contour_legend:
        handles = [
            Patch(facecolor=color, alpha=0.85, label=label)
            for color, label in zip(
                contour_palette,
                _contour_labels(resolved_contour_levels),
                strict=True,
            )
        ]
        ax.legend(handles=handles, title="Two-sided significance", loc="best")
    if created_axes:
        ax.figure.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.96)
    return ax
