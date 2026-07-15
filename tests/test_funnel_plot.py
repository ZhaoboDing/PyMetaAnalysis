"""Funnel plot behavior tested with Matplotlib's non-interactive backend."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from scipy.stats import norm  # noqa: E402

import meta_analyze as ma  # noqa: E402


def _generic_result(*, model: str = "common") -> ma.MetaAnalysisResult:
    effect = np.linspace(-0.35, 0.55, 10)
    variance = np.linspace(0.015, 0.12, 10)
    return ma.meta_analysis(
        effect=effect,
        variance=variance,
        study=[f"Study {index}" for index in range(1, 11)],
        model=model,
        tau2_method="PM",
    )


def _scatter(axes: Any) -> PathCollection:
    return next(
        collection
        for collection in axes.collections
        if isinstance(collection, PathCollection)
    )


def test_identity_funnel_coordinates_and_pseudo_limits() -> None:
    result = _generic_result()
    axes = result.funnel(warn_on_few_studies=False)
    scatter = _scatter(axes)

    assert axes.get_xscale() == "linear"
    assert axes.yaxis_inverted()
    assert axes.get_xlabel() == "Effect"
    assert axes.get_ylabel() == "Standard error"
    np.testing.assert_allclose(
        scatter.get_offsets()[:, 0], result.study_results["effect"]
    )
    np.testing.assert_allclose(
        scatter.get_offsets()[:, 1], np.sqrt(result.study_results["variance"])
    )
    np.testing.assert_allclose(scatter.get_sizes(), [42.0])

    lower_line, upper_line, reference_line = axes.lines
    standard_error_grid = np.asarray(lower_line.get_ydata())
    critical_value = norm.ppf(0.975)
    np.testing.assert_allclose(
        lower_line.get_xdata(), result.estimate - critical_value * standard_error_grid
    )
    np.testing.assert_allclose(
        upper_line.get_xdata(), result.estimate + critical_value * standard_error_grid
    )
    np.testing.assert_allclose(
        reference_line.get_xdata(), [result.estimate, result.estimate]
    )
    plt.close(axes.figure)


def test_ratio_funnel_uses_exponentiated_log_axis() -> None:
    result = ma.meta_binary(
        event_treat=[12, 5, 20, 7, 3, 9, 14, 6, 11, 8],
        n_treat=[100, 80, 120, 90, 55, 70, 95, 60, 88, 72],
        event_control=[18, 9, 15, 10, 7, 12, 16, 9, 14, 11],
        n_control=[110, 75, 130, 95, 60, 74, 100, 64, 92, 76],
        measure="RR",
        method="MH",
    )
    axes = result.funnel(warn_on_few_studies=False)
    scatter = _scatter(axes)

    assert axes.get_xscale() == "log"
    assert axes.get_xlabel() == "Risk ratio"
    assert axes.xaxis.get_major_formatter()(0.5, 0) == "0.5"
    np.testing.assert_allclose(
        scatter.get_offsets()[:, 0], np.exp(result.study_results["effect"])
    )
    np.testing.assert_allclose(
        axes.lines[-1].get_xdata(),
        [result.display_estimate, result.display_estimate],
    )
    plt.close(axes.figure)


def test_random_funnel_limits_do_not_include_tau2() -> None:
    result = _generic_result(model="random")
    assert result.tau2 > 0.0
    axes = result.funnel(
        confidence_level=0.90,
        warn_on_few_studies=False,
    )

    lower_line = axes.lines[0]
    standard_error_grid = np.asarray(lower_line.get_ydata())
    expected = result.estimate - norm.ppf(0.95) * standard_error_grid
    np.testing.assert_allclose(lower_line.get_xdata(), expected)
    plt.close(axes.figure)


def test_pseudo_confidence_region_can_be_hidden() -> None:
    result = _generic_result()
    axes = result.funnel(
        show_pseudo_confidence_interval=False,
        warn_on_few_studies=False,
    )

    assert len(axes.lines) == 1
    assert len(axes.collections) == 1
    plt.close(axes.figure)


def test_funnel_warns_for_fewer_than_ten_studies_and_omits_exclusions() -> None:
    result = ma.meta_analysis(
        effect=[0.1, np.nan, 0.3],
        variance=[0.01, 0.02, 0.04],
        model="common",
        missing="drop",
    )
    with pytest.warns(UserWarning, match="fewer than 10"):
        axes = result.funnel()

    assert len(_scatter(axes).get_offsets()) == 2
    plt.close(axes.figure)


@pytest.mark.parametrize("level", [0.0, 1.0, np.nan, "95%"])
def test_invalid_funnel_confidence_level(level: object) -> None:
    result = _generic_result()
    with pytest.raises(ValueError, match="between 0 and 1"):
        result.funnel(
            confidence_level=level,  # type: ignore[arg-type]
            warn_on_few_studies=False,
        )


def test_identity_funnel_rejects_incompatible_log_axis() -> None:
    result = _generic_result()
    with pytest.raises(ValueError, match="strictly positive"):
        result.funnel(log_scale=True, warn_on_few_studies=False)


def test_funnel_does_not_call_show_and_can_render_png(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_show(*args: object, **kwargs: object) -> None:
        raise AssertionError("funnel() must not call show()")

    monkeypatch.setattr(plt, "show", fail_show)
    result = _generic_result()
    figure, axes = plt.subplots()
    returned = result.funnel(
        ax=axes,
        effect_label="Treatment effect",
        warn_on_few_studies=False,
    )
    output = tmp_path / "funnel.png"
    returned.figure.savefig(output, dpi=100)

    assert returned is axes
    assert returned.get_xlabel() == "Treatment effect"
    assert output.stat().st_size > 1000
    plt.close(figure)


def test_missing_plot_extra_has_actionable_funnel_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _generic_result()
    real_import = builtins.__import__

    def reject_matplotlib(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "matplotlib.pyplot":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_matplotlib)
    with pytest.raises(ImportError, match=r"PyMetaAnalysis\[plot\]"):
        result.funnel(warn_on_few_studies=False)
