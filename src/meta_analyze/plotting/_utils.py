"""Shared display-scale helpers for optional plotting backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..results import MetaAnalysisResult
else:
    Axes = Any


def to_display_scale(
    values: NDArray[np.float64], *, display_scale: str
) -> NDArray[np.float64]:
    """Transform model-scale values to a result's display scale."""

    if display_scale == "identity":
        displayed = values.copy()
    elif display_scale == "exp":
        with np.errstate(over="ignore"):
            displayed = np.exp(values)
    else:
        raise ValueError(f"Unknown display scale {display_scale!r}.")
    if not np.all(np.isfinite(displayed)):
        raise ValueError(
            "Plot values are non-finite after display-scale transformation."
        )
    return displayed


def default_effect_label(result: MetaAnalysisResult) -> str:
    """Return a readable x-axis label for a fitted measure."""

    labels = {
        "OR": "Odds ratio",
        "RR": "Risk ratio",
        "RD": "Risk difference",
        "MD": "Mean difference",
        "SMD": "Standardized mean difference",
        "GENERIC": "Effect",
    }
    return labels.get(result.measure, result.measure)


def configure_log_axis(ax: Axes) -> None:
    """Use readable ratio ticks on a logarithmic x-axis."""

    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def marker_areas(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale positive relative weights to readable plot marker areas."""

    largest = float(np.max(weights))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("Plot weights must be finite and strictly positive.")
    return 24.0 + 176.0 * weights / largest
