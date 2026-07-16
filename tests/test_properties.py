from __future__ import annotations

import numpy as np
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
