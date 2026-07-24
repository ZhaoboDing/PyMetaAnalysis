from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import meta_analyze as ma


@st.composite
def study_vectors(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    size = draw(st.integers(min_value=2, max_value=20))
    effects = draw(
        st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=size,
            max_size=size,
        )
    )
    variances = draw(
        st.lists(
            st.floats(
                min_value=1e-4,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=size,
            max_size=size,
        )
    )
    return np.asarray(effects), np.asarray(variances)


@st.composite
def meta_regression_vectors(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = draw(st.integers(min_value=5, max_value=20))
    effects = draw(
        st.lists(
            st.floats(
                min_value=-5.0,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=size,
            max_size=size,
        )
    )
    variances = draw(
        st.lists(
            st.floats(
                min_value=1e-3,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=size,
            max_size=size,
        )
    )
    moderator = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    return np.asarray(effects), np.asarray(variances), moderator


@given(study_vectors())
@settings(max_examples=75, deadline=None)
def test_common_model_is_invariant_to_row_order(
    vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    effect, variance = vectors
    forward = ma.meta_analysis(effect=effect, variance=variance, model="common")
    order = np.arange(len(effect))[::-1]
    reverse = ma.meta_analysis(
        effect=effect[order], variance=variance[order], model="common"
    )

    assert reverse.estimate == pytest.approx(forward.estimate, rel=1e-12, abs=1e-12)
    assert reverse.standard_error == pytest.approx(
        forward.standard_error, rel=1e-12, abs=1e-12
    )
    assert reverse.q == pytest.approx(forward.q, rel=1e-11, abs=1e-11)


@given(study_vectors(), st.floats(min_value=-20, max_value=20))
@settings(max_examples=75, deadline=None)
def test_common_model_is_translation_equivariant(
    vectors: tuple[np.ndarray, np.ndarray], shift: float
) -> None:
    effect, variance = vectors
    original = ma.meta_analysis(effect=effect, variance=variance, model="common")
    shifted = ma.meta_analysis(effect=effect + shift, variance=variance, model="common")

    assert shifted.estimate == pytest.approx(original.estimate + shift, abs=1e-10)
    assert shifted.standard_error == pytest.approx(original.standard_error, abs=1e-12)
    assert shifted.q == pytest.approx(original.q, rel=1e-10, abs=1e-10)


@given(study_vectors())
@settings(max_examples=75, deadline=None)
def test_normalized_weights_are_nonnegative_and_sum_to_one(
    vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    effect, variance = vectors
    result = ma.meta_analysis(effect=effect, variance=variance, model="common")
    weights = result.study_results["normalized_weight"].to_numpy()

    assert np.all(weights >= 0.0)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-12)


@given(study_vectors())
@settings(max_examples=75, deadline=None)
def test_standard_error_and_variance_inputs_are_equivalent(
    vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    effect, variance = vectors
    from_variance = ma.meta_analysis(
        effect=effect,
        variance=variance,
        model="common",
    )
    from_standard_error = ma.meta_analysis(
        effect=effect,
        standard_error=np.sqrt(variance),
        model="common",
    )

    assert from_standard_error.estimate == pytest.approx(
        from_variance.estimate, rel=1e-12, abs=1e-12
    )
    assert from_standard_error.standard_error == pytest.approx(
        from_variance.standard_error, rel=1e-12, abs=1e-12
    )
    assert from_standard_error.q == pytest.approx(from_variance.q, rel=1e-11, abs=1e-11)
    np.testing.assert_allclose(
        from_standard_error.study_results["normalized_weight"],
        from_variance.study_results["normalized_weight"],
        rtol=1e-12,
        atol=1e-12,
    )


@given(meta_regression_vectors())
@settings(max_examples=50, deadline=None)
def test_common_meta_regression_is_invariant_to_row_order(
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    effect, variance, moderator = vectors
    forward = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        model="common",
    )
    order = np.arange(len(effect))[::-1]
    reverse = ma.meta_regression(
        effect=effect[order],
        variance=variance[order],
        moderators={"x": moderator[order]},
        model="common",
    )

    np.testing.assert_allclose(
        reverse.coefficients["estimate"],
        forward.coefficients["estimate"],
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        reverse.coefficient_covariance,
        forward.coefficient_covariance,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        reverse.study_results["fitted_value"].to_numpy()[::-1],
        forward.study_results["fitted_value"],
        rtol=1e-11,
        atol=1e-11,
    )
    assert reverse.heterogeneity.q == pytest.approx(
        forward.heterogeneity.q, rel=1e-10, abs=1e-10
    )
    assert reverse.global_test.statistic == pytest.approx(
        forward.global_test.statistic, rel=1e-10, abs=1e-10
    )


@given(
    meta_regression_vectors(),
    st.floats(min_value=-20.0, max_value=20.0),
    st.floats(min_value=0.2, max_value=5.0),
)
@settings(max_examples=50, deadline=None)
def test_common_meta_regression_is_equivariant_to_location_and_scale(
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    effect_shift: float,
    moderator_scale: float,
) -> None:
    effect, variance, moderator = vectors
    original = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        model="common",
    )
    transformed = ma.meta_regression(
        effect=effect + effect_shift,
        variance=variance,
        moderators={"x": moderator * moderator_scale},
        model="common",
    )

    original_coefficients = original.coefficients.set_index("term")["estimate"]
    transformed_coefficients = transformed.coefficients.set_index("term")["estimate"]
    assert transformed_coefficients["intercept"] == pytest.approx(
        original_coefficients["intercept"] + effect_shift,
        rel=1e-10,
        abs=1e-10,
    )
    assert transformed_coefficients["x"] == pytest.approx(
        original_coefficients["x"] / moderator_scale,
        rel=1e-10,
        abs=1e-10,
    )
    np.testing.assert_allclose(
        transformed.study_results["fitted_value"] - effect_shift,
        original.study_results["fitted_value"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        transformed.study_results["residual"],
        original.study_results["residual"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert transformed.heterogeneity.q == pytest.approx(
        original.heterogeneity.q, rel=1e-10, abs=1e-10
    )
    assert transformed.global_test.statistic == pytest.approx(
        original.global_test.statistic, rel=1e-10, abs=1e-10
    )


@given(meta_regression_vectors())
@settings(max_examples=50, deadline=None)
def test_riley_prediction_intervals_are_symmetric_and_wider_than_default(
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    effect, variance, moderator = vectors
    default = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        model="mixed",
        tau2_method="DL",
    )
    riley = ma.meta_regression(
        effect=effect,
        variance=variance,
        moderators={"x": moderator},
        model="mixed",
        tau2_method="DL",
        prediction_interval_method="riley",
    )
    prediction_points = np.asarray([-0.75, 0.25, 0.9])
    default_prediction = default.predict(pd.DataFrame({"x": prediction_points}))
    riley_prediction = riley.predict(pd.DataFrame({"x": prediction_points}))

    np.testing.assert_allclose(
        riley_prediction[["estimate", "standard_error", "ci_low", "ci_high"]],
        default_prediction[["estimate", "standard_error", "ci_low", "ci_high"]],
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        (riley_prediction["pi_low"] + riley_prediction["pi_high"]) / 2.0,
        riley_prediction["estimate"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(
        riley_prediction["pi_high"] - riley_prediction["pi_low"]
        > default_prediction["pi_high"] - default_prediction["pi_low"]
    )
