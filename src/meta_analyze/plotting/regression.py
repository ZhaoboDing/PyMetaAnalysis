"""Matplotlib bubble plots for single-moderator meta-regression results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..exceptions import UnsupportedMethodError
from ..regression_results import MetaRegressionResult
from ._utils import marker_areas

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def _numeric_moderator_name(result: MetaRegressionResult) -> str:
    moderators = result.design_info.moderators
    if not result.design_info.intercept:
        raise UnsupportedMethodError(
            "Bubble plots require a meta-regression fitted with an intercept."
        )
    if len(moderators) != 1 or moderators[0].kind != "numeric":
        raise UnsupportedMethodError(
            "Bubble plots require exactly one numeric moderator; multivariable "
            "and categorical marginal plots are not inferred automatically."
        )
    return moderators[0].name


def bubble_plot(
    result: MetaRegressionResult,
    *,
    ax: Axes | None = None,
    moderator_label: str | None = None,
    effect_label: str = "Effect",
    show_confidence_interval: bool = True,
    show_prediction_interval: bool = False,
) -> Axes:
    """Draw a weighted single-moderator bubble plot and return its axes.

    Marker area is proportional to normalized fitted precision weight. The
    fitted line and optional intervals are generated through ``result.predict``
    so they use the model's stored encoding, covariance, and inference method.
    The function never calls ``show()``.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - tested without plot extra
        raise ImportError(
            "Bubble plots require Matplotlib; install PyMetaAnalysis[plot]."
        ) from error

    moderator = _numeric_moderator_name(result)
    if show_prediction_interval and result.model != "mixed":
        raise UnsupportedMethodError(
            "Prediction intervals are only available for mixed-effects "
            "meta-regression bubble plots."
        )

    studies = result.study_results
    included = studies.loc[studies["included"]].reset_index(drop=True)
    x_values = included[moderator].to_numpy(dtype=np.float64, copy=True)
    effects = included["effect"].to_numpy(dtype=np.float64, copy=True)
    weights = included["normalized_precision_weight"].to_numpy(
        dtype=np.float64, copy=True
    )
    grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 200)
    predictions = result.predict(pd.DataFrame({moderator: grid}))

    created_axes = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 5.0))

    if show_prediction_interval:
        ax.fill_between(
            grid,
            predictions["pi_low"].to_numpy(dtype=np.float64),
            predictions["pi_high"].to_numpy(dtype=np.float64),
            color="#fde7c7",
            alpha=0.55,
            label="Prediction interval",
            zorder=0,
        )
    if show_confidence_interval:
        ax.fill_between(
            grid,
            predictions["ci_low"].to_numpy(dtype=np.float64),
            predictions["ci_high"].to_numpy(dtype=np.float64),
            color="#dbeafe",
            alpha=0.75,
            label="Mean confidence interval",
            zorder=1,
        )
    ax.plot(
        grid,
        predictions["estimate"].to_numpy(dtype=np.float64),
        color="#1f4e79",
        linewidth=2.0,
        label="Fitted mean",
        zorder=2,
    )
    ax.scatter(
        x_values,
        effects,
        s=marker_areas(weights),
        color="#2f6f9f",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.9,
        zorder=3,
    )
    ax.set_xlabel(moderator_label or moderator)
    ax.set_ylabel(effect_label)
    ax.grid(color="#e5e7eb", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.05, y=0.08)
    if show_confidence_interval or show_prediction_interval:
        ax.legend(frameon=False)
    if created_axes:
        ax.figure.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.96)
    return ax
