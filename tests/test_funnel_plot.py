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
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import PathCollection, PolyCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
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
        tau2_method="PM" if model == "random" else None,
    )


def _scatter(axes: Any) -> PathCollection:
    return next(
        collection
        for collection in axes.collections
        if isinstance(collection, PathCollection)
    )


def _filled_regions(axes: Any) -> list[PolyCollection]:
    return [
        collection
        for collection in axes.collections
        if isinstance(collection, PolyCollection)
    ]


def _capture_fill_betweenx(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    original = Axes.fill_betweenx

    def capture(
        axes: Axes,
        y: Any,
        x1: Any,
        x2: Any,
        *args: Any,
        **kwargs: Any,
    ) -> PolyCollection:
        calls.append(
            (
                np.asarray(y, dtype=float).copy(),
                np.asarray(x1, dtype=float).copy(),
                np.asarray(x2, dtype=float).copy(),
            )
        )
        return original(axes, y, x1, x2, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_betweenx", capture)
    return calls


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


def test_correlation_funnel_back_transforms_fisher_z() -> None:
    correlations = np.linspace(-0.45, 0.65, 10)
    result = ma.meta_correlation(
        correlation=correlations,
        n=np.arange(20, 120, 10),
        model="common",
    )
    axes = result.funnel(warn_on_few_studies=False)
    scatter = _scatter(axes)

    assert axes.get_xscale() == "linear"
    assert axes.get_xlabel() == "Correlation"
    np.testing.assert_allclose(scatter.get_offsets()[:, 0], correlations)
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


def test_contour_regions_match_two_sided_identity_scale_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _generic_result()
    levels = (0.90, 0.95, 0.99)
    fills = _capture_fill_betweenx(monkeypatch)
    axes = result.funnel(
        contour_levels=levels,
        warn_on_few_studies=False,
    )
    regions = _filled_regions(axes)
    index = 100
    y = float(fills[0][0][index])

    assert len(regions) == 2 * len(levels)
    assert len(fills) == 2 * len(levels)
    assert len(axes.lines) == 4
    for level_index, level in enumerate(levels):
        critical = norm.ppf(0.5 + level / 2.0)
        assert fills[2 * level_index][2][index] == pytest.approx(-critical * y)
        assert fills[2 * level_index + 1][1][index] == pytest.approx(critical * y)

    null_line = next(line for line in axes.lines if line.get_linestyle() == ":")
    np.testing.assert_allclose(null_line.get_xdata(), [0.0, 0.0])
    legend = axes.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "Two-sided significance"
    assert [text.get_text() for text in legend.get_texts()] == [
        "0.05 < p <= 0.1",
        "0.01 < p <= 0.05",
        "p <= 0.01",
    ]
    brightness = [float(np.mean(region.get_facecolor()[0, :3])) for region in regions]
    assert brightness[0] > brightness[2] > brightness[4]
    plt.close(axes.figure)


def test_ratio_contours_use_null_one_and_log_transformed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ma.meta_binary(
        event_treat=[12, 5, 20, 7],
        n_treat=[100, 80, 120, 90],
        event_control=[18, 9, 15, 10],
        n_control=[110, 75, 130, 95],
        measure="OR",
        method="MH",
    )
    fills = _capture_fill_betweenx(monkeypatch)
    axes = result.funnel(
        contour_levels=(0.95,),
        show_pseudo_confidence_interval=False,
        warn_on_few_studies=False,
    )
    index = 75
    y = float(fills[0][0][index])
    critical = norm.ppf(0.975)

    assert axes.get_xscale() == "log"
    assert len(fills) == 2
    assert fills[0][2][index] == pytest.approx(np.exp(-critical * y))
    assert fills[1][1][index] == pytest.approx(np.exp(critical * y))
    null_line = next(line for line in axes.lines if line.get_linestyle() == ":")
    np.testing.assert_allclose(null_line.get_xdata(), [1.0, 1.0])
    assert axes.get_xlim()[0] > 0.0
    plt.close(axes.figure)


def test_correlation_contours_transform_custom_display_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ma.meta_correlation(
        correlation=np.linspace(-0.45, 0.65, 10),
        n=np.arange(20, 120, 10),
        model="common",
    )
    reference = 0.2
    fills = _capture_fill_betweenx(monkeypatch)
    axes = result.funnel(
        contour_levels=(0.90,),
        contour_reference=reference,
        show_pseudo_confidence_interval=False,
        warn_on_few_studies=False,
    )
    index = 75
    y = float(fills[0][0][index])
    critical = norm.ppf(0.95)
    model_reference = np.arctanh(reference)

    assert fills[0][2][index] == pytest.approx(np.tanh(model_reference - critical * y))
    assert fills[1][1][index] == pytest.approx(np.tanh(model_reference + critical * y))
    null_line = next(line for line in axes.lines if line.get_linestyle() == ":")
    np.testing.assert_allclose(null_line.get_xdata(), [reference, reference])
    plt.close(axes.figure)


def test_contour_colors_and_legend_can_be_configured() -> None:
    result = _generic_result()
    colors = ("#fee2e2", "#ef4444")
    axes = result.funnel(
        contour_levels=(0.90, 0.95),
        contour_colors=colors,
        show_contour_legend=False,
        warn_on_few_studies=False,
    )
    regions = _filled_regions(axes)

    np.testing.assert_allclose(
        regions[0].get_facecolor()[0], (*to_rgba(colors[0])[:3], 0.85)
    )
    np.testing.assert_allclose(
        regions[2].get_facecolor()[0], (*to_rgba(colors[1])[:3], 0.85)
    )
    assert axes.get_legend() is None
    plt.close(axes.figure)


@pytest.mark.parametrize(
    "levels",
    [
        (),
        0.95,
        "0.95",
        (0.0,),
        (1.0,),
        (np.nan,),
        (True,),
        ("0.95",),
        (0.95, 0.90),
        (0.90, 0.90),
    ],
)
def test_invalid_contour_levels(levels: object) -> None:
    result = _generic_result()
    with pytest.raises(ValueError, match="contour"):
        result.funnel(
            contour_levels=levels,  # type: ignore[arg-type]
            warn_on_few_studies=False,
        )


@pytest.mark.parametrize(
    "colors",
    [42, "gray", ("gray",), ("gray", "not-a-color")],
)
def test_invalid_contour_colors(colors: object) -> None:
    result = _generic_result()
    with pytest.raises(ValueError, match="contour"):
        result.funnel(
            contour_levels=(0.90, 0.95),
            contour_colors=colors,  # type: ignore[arg-type]
            warn_on_few_studies=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"contour_colors": ("gray",)}, {"contour_reference": 0.0}],
)
def test_contour_options_require_levels(kwargs: dict[str, object]) -> None:
    result = _generic_result()
    with pytest.raises(ValueError, match="require contour_levels"):
        result.funnel(warn_on_few_studies=False, **kwargs)  # type: ignore[arg-type]


def test_invalid_contour_reference_for_display_scale() -> None:
    ratio = ma.meta_binary(
        event_treat=[3, 5, 7],
        n_treat=[40, 50, 60],
        event_control=[6, 8, 10],
        n_control=[42, 52, 62],
        measure="OR",
    )
    correlation = ma.meta_correlation(
        correlation=[-0.2, 0.1, 0.3],
        n=[30, 40, 50],
        model="common",
    )

    with pytest.raises(ValueError, match="strictly positive"):
        ratio.funnel(
            contour_levels=(0.95,),
            contour_reference=0.0,
            warn_on_few_studies=False,
        )
    with pytest.raises(ValueError, match="between -1 and 1"):
        correlation.funnel(
            contour_levels=(0.95,),
            contour_reference=1.0,
            warn_on_few_studies=False,
        )
    with pytest.raises(ValueError, match="finite"):
        correlation.funnel(
            contour_levels=(0.95,),
            contour_reference=np.nan,
            warn_on_few_studies=False,
        )


def test_contour_limits_must_be_positive_on_manual_log_axis() -> None:
    result = ma.meta_analysis(
        effect=[0.4, 0.6, 0.8],
        variance=[0.04, 0.05, 0.06],
        model="common",
    )

    with pytest.raises(ValueError, match="contour limits.*strictly positive"):
        result.funnel(
            contour_levels=(0.95,),
            contour_reference=0.1,
            show_pseudo_confidence_interval=False,
            log_scale=True,
            warn_on_few_studies=False,
        )


def test_invalid_contour_legend_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        _generic_result().funnel(
            contour_levels=(0.95,),
            show_contour_legend="yes",  # type: ignore[arg-type]
            warn_on_few_studies=False,
        )


def test_invalid_contour_configuration_does_not_mutate_supplied_axes() -> None:
    figure, axes = plt.subplots()

    with pytest.raises(ValueError, match="Matplotlib color"):
        _generic_result().funnel(
            ax=axes,
            contour_levels=(0.90,),
            contour_colors=("not-a-color",),
            warn_on_few_studies=False,
        )

    assert not axes.lines
    assert not axes.collections
    assert axes.get_xscale() == "linear"
    plt.close(figure)


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
        contour_levels=(0.90, 0.95, 0.99),
        warn_on_few_studies=False,
    )
    output = tmp_path / "funnel.png"
    returned.figure.savefig(output, dpi=100)

    assert returned is axes
    assert returned.get_xlabel() == "Treatment effect"
    assert returned.get_legend() is not None
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
