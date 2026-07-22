"""Meta-regression bubble plots tested with a non-interactive backend."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection, PolyCollection  # noqa: E402

import meta_analyze as ma  # noqa: E402


def _numeric_result(*, model: str = "mixed") -> ma.MetaRegressionResult:
    moderator = np.linspace(0.0, 4.5, 10)
    effect = (
        0.15
        + 0.28 * moderator
        + np.asarray([-0.20, 0.08, -0.12, 0.22, -0.04, 0.18, -0.16, 0.25, -0.08, 0.12])
    )
    return ma.meta_regression(
        effect=effect,
        variance=np.linspace(0.025, 0.10, 10),
        moderators={"dose": moderator},
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


def test_bubble_coordinates_sizes_fit_and_confidence_band() -> None:
    result = _numeric_result()
    axes = result.bubble(moderator_label="Dose", effect_label="Mean difference")
    scatter = _scatter(axes)
    studies = result.study_results

    np.testing.assert_allclose(scatter.get_offsets()[:, 0], studies["dose"])
    np.testing.assert_allclose(scatter.get_offsets()[:, 1], studies["effect"])
    weights = studies["normalized_precision_weight"].to_numpy()
    assert np.array_equal(np.argsort(scatter.get_sizes()), np.argsort(weights))

    fitted_line = axes.lines[0]
    line_x = np.asarray(fitted_line.get_xdata(), dtype=np.float64)
    expected = result.predict(pd.DataFrame({"dose": line_x}))
    np.testing.assert_allclose(fitted_line.get_ydata(), expected["estimate"])
    assert axes.get_xlabel() == "Dose"
    assert axes.get_ylabel() == "Mean difference"
    assert sum(isinstance(item, PolyCollection) for item in axes.collections) == 1
    assert {text.get_text() for text in axes.get_legend().get_texts()} == {
        "Fitted mean",
        "Mean confidence interval",
    }
    plt.close(axes.figure)


def test_mixed_bubble_can_show_prediction_and_confidence_intervals() -> None:
    result = _numeric_result()
    axes = result.bubble(show_prediction_interval=True)

    assert sum(isinstance(item, PolyCollection) for item in axes.collections) == 2
    assert {text.get_text() for text in axes.get_legend().get_texts()} == {
        "Fitted mean",
        "Mean confidence interval",
        "Prediction interval",
    }
    plt.close(axes.figure)


def test_bubble_bands_can_be_hidden() -> None:
    result = _numeric_result()
    axes = result.bubble(
        show_confidence_interval=False,
        show_prediction_interval=False,
    )

    assert len(axes.collections) == 1
    assert axes.get_legend() is None
    plt.close(axes.figure)


def test_bubble_omits_excluded_rows() -> None:
    moderator = np.linspace(0.0, 5.0, 11)
    effect = 0.2 + 0.15 * moderator
    effect[4] = np.nan
    result = ma.meta_regression(
        effect=effect,
        variance=np.linspace(0.03, 0.08, 11),
        moderators={"dose": moderator},
        model="common",
        missing="drop",
    )
    axes = result.bubble()

    assert len(_scatter(axes).get_offsets()) == 10
    assert result.excluded_studies["row_id"].tolist() == [4]
    plt.close(axes.figure)


def test_bubble_rejects_ambiguous_model_shapes() -> None:
    data = pd.DataFrame(
        {
            "yi": [0.1, 0.3, 0.4, 0.7, 0.9, 1.0],
            "vi": [0.04, 0.05, 0.06, 0.04, 0.05, 0.06],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    multiple = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["x", "z"],
        model="common",
    )
    categorical = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["group"],
        categorical={"group": ["A", "B"]},
        model="common",
    )
    no_intercept = ma.meta_regression(
        data,
        effect="yi",
        variance="vi",
        moderators=["x"],
        intercept=False,
        model="common",
    )

    with pytest.raises(ma.UnsupportedMethodError, match="exactly one numeric"):
        multiple.bubble()
    with pytest.raises(ma.UnsupportedMethodError, match="exactly one numeric"):
        categorical.bubble()
    with pytest.raises(ma.UnsupportedMethodError, match="fitted with an intercept"):
        no_intercept.bubble()


def test_common_bubble_rejects_prediction_interval_request() -> None:
    result = _numeric_result(model="common")
    with pytest.raises(ma.UnsupportedMethodError, match="only available for mixed"):
        result.bubble(show_prediction_interval=True)


def test_bubble_reuses_axes_does_not_show_and_can_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_show(*args: object, **kwargs: object) -> None:
        raise AssertionError("bubble() must not call show()")

    monkeypatch.setattr(plt, "show", fail_show)
    result = _numeric_result()
    figure, axes = plt.subplots()
    returned = result.bubble(ax=axes)
    output = tmp_path / "bubble.png"
    returned.figure.savefig(output, dpi=100)

    assert returned is axes
    assert output.stat().st_size > 1000
    plt.close(figure)


def test_missing_plot_extra_has_actionable_bubble_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _numeric_result()
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
        result.bubble()
