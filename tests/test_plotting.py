"""Forest plot behavior tested with Matplotlib's non-interactive backend."""

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

import meta_analyze as ma  # noqa: E402


def test_identity_forest_contains_studies_weights_and_pooled_diamond() -> None:
    result = ma.meta_analysis(
        effect=[0.1, 0.35, -0.15],
        variance=[0.01, 0.04, 0.09],
        study=["Alpha", "Beta", "Gamma"],
        model="common",
    )
    figure, axes = plt.subplots()
    returned = result.forest(ax=axes, effect_label="Mean change")

    assert returned is axes
    assert axes.get_xscale() == "linear"
    assert axes.get_xlabel() == "Mean change"
    assert [tick.get_text() for tick in axes.get_yticklabels()] == [
        "Alpha",
        "Beta",
        "Gamma",
        "Common effect",
    ]
    assert len(axes.patches) == 1
    diamond = axes.patches[0].get_xy()
    np.testing.assert_allclose(
        diamond[:4, 0],
        [result.ci_low, result.estimate, result.ci_high, result.estimate],
    )

    scatter = next(
        collection
        for collection in axes.collections
        if isinstance(collection, PathCollection)
    )
    marker_areas = scatter.get_sizes()
    weights = result.study_results["normalized_weight"].to_numpy()
    assert np.array_equal(np.argsort(marker_areas), np.argsort(weights))
    assert "Weight" in {text.get_text() for text in axes.texts}
    assert {f"{100 * weight:.1f}%" for weight in weights}.issubset(
        {text.get_text() for text in axes.texts}
    )
    plt.close(figure)


def test_ratio_forest_uses_display_scale_log_axis_and_null_one() -> None:
    result = ma.meta_binary(
        event_treat=[12, 5, 20],
        n_treat=[100, 80, 120],
        event_control=[18, 9, 15],
        n_control=[110, 75, 130],
        measure="RR",
        method="MH",
    )
    axes = result.forest(show_weights=False)

    assert axes.get_xscale() == "log"
    assert axes.get_xlabel() == "Risk ratio"
    null_line = axes.lines[0]
    np.testing.assert_allclose(null_line.get_xdata(), [1.0, 1.0])
    diamond = axes.patches[0].get_xy()
    np.testing.assert_allclose(
        diamond[:4, 0],
        [
            result.display_ci[0],
            result.display_estimate,
            result.display_ci[1],
            result.display_estimate,
        ],
    )
    assert not axes.texts
    plt.close(axes.figure)


def test_prediction_interval_can_be_shown_or_hidden() -> None:
    result = ma.meta_analysis(
        effect=[-0.2, 0.1, 0.5, 0.8],
        variance=[0.03, 0.02, 0.04, 0.05],
        model="random",
        tau2_method="PM",
    )
    assert result.prediction_interval is not None

    with_prediction = result.forest(show_prediction_interval=True)
    without_prediction = result.forest(show_prediction_interval=False)

    assert len(with_prediction.collections) == len(without_prediction.collections) + 2
    assert with_prediction.get_yticklabels()[-1].get_text() == "Random effects"
    plt.close(with_prediction.figure)
    plt.close(without_prediction.figure)


def test_forest_omits_excluded_rows_but_preserves_result() -> None:
    result = ma.meta_analysis(
        effect=[0.1, np.nan, 0.3],
        variance=[0.01, 0.02, 0.03],
        study=["Included A", "Excluded", "Included B"],
        model="common",
        missing="drop",
    )
    axes = result.forest(show_weights=False)

    labels = [tick.get_text() for tick in axes.get_yticklabels()]
    assert labels == ["Included A", "Included B", "Common effect"]
    assert result.excluded_studies["study"].tolist() == ["Excluded"]
    plt.close(axes.figure)


def test_forest_does_not_call_show_and_can_render_png(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_show(*args: object, **kwargs: object) -> None:
        raise AssertionError("forest() must not call show()")

    monkeypatch.setattr(plt, "show", fail_show)
    result = ma.meta_continuous(
        mean_treat=[2.0, 3.0],
        sd_treat=[1.0, 1.2],
        n_treat=[20, 25],
        mean_control=[1.5, 2.2],
        sd_control=[1.1, 1.0],
        n_control=[18, 24],
        measure="SMD",
        model="common",
    )
    axes = result.forest(pooled_label="Overall Hedges' g")
    output = tmp_path / "forest.png"
    axes.figure.savefig(output, dpi=100)

    assert output.stat().st_size > 1000
    assert axes.get_yticklabels()[-1].get_text() == "Overall Hedges' g"
    plt.close(axes.figure)


def test_log_axis_rejects_nonpositive_null_value() -> None:
    result = ma.meta_binary(
        event_treat=[2, 3],
        n_treat=[20, 20],
        event_control=[3, 4],
        n_control=[20, 20],
        measure="OR",
        method="MH",
    )
    with pytest.raises(ValueError, match="strictly positive"):
        result.forest(null_value=0.0)


def test_missing_plot_extra_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ma.meta_analysis(effect=[0.1], variance=[0.02], model="common")
    real_import = builtins.__import__

    def reject_matplotlib(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name in {"matplotlib.pyplot", "matplotlib.patches"}:
            raise ImportError("simulated missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_matplotlib)
    with pytest.raises(ImportError, match=r"PyMetaAnalysis\[plot\]"):
        result.forest()
