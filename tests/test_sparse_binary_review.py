"""Independent formula checks for sparse-binary common-effect estimators."""

from __future__ import annotations

import json
import math
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

import meta_analyze as ma
from meta_analyze.estimators import fit_mantel_haenszel, fit_peto

R_REFERENCE = json.loads(
    (
        Path(__file__).parent / "reference" / "sparse_binary_review_metafor.json"
    ).read_text(encoding="utf-8")
)
REFERENCE_RTOL = 5e-13
REFERENCE_ATOL = 5e-15


class FormulaFit(NamedTuple):
    estimate: float
    standard_error: float
    normalized_weights: np.ndarray
    q: float | None = None


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _log_fraction(value: Fraction) -> float:
    if value <= 0:
        raise ValueError("A logarithm requires a positive fraction.")
    difference = value.numerator - value.denominator
    if abs(difference) <= min(value.numerator, value.denominator):
        return math.log1p(float(Fraction(difference, value.denominator)))
    return math.log(value.numerator) - math.log(value.denominator)


def _normalize(weights: list[Fraction]) -> np.ndarray:
    total = sum(weights, start=Fraction())
    return np.asarray([float(weight / total) for weight in weights])


def _mh_formula(
    a_values: np.ndarray,
    b_values: np.ndarray,
    c_values: np.ndarray,
    d_values: np.ndarray,
    measure: str,
) -> FormulaFit:
    rows = [
        tuple(_fraction(value) for value in row)
        for row in zip(a_values, b_values, c_values, d_values, strict=True)
    ]
    if measure == "OR":
        r = sum((a * d / (a + b + c + d) for a, b, c, d in rows), Fraction())
        s = sum((b * c / (a + b + c + d) for a, b, c, d in rows), Fraction())
        e = sum(
            ((a + d) * a * d / (a + b + c + d) ** 2 for a, b, c, d in rows),
            Fraction(),
        )
        f = sum(
            ((a + d) * b * c / (a + b + c + d) ** 2 for a, b, c, d in rows),
            Fraction(),
        )
        g = sum(
            ((b + c) * a * d / (a + b + c + d) ** 2 for a, b, c, d in rows),
            Fraction(),
        )
        h = sum(
            ((b + c) * b * c / (a + b + c + d) ** 2 for a, b, c, d in rows),
            Fraction(),
        )
        variance = (e / r**2 + (f + g) / (r * s) + h / s**2) / 2
        weights = [b * c / (a + b + c + d) for a, b, c, d in rows]
        return FormulaFit(
            _log_fraction(r / s),
            math.sqrt(float(variance)),
            _normalize(weights),
        )

    if measure == "RR":
        r = sum(
            (a * (c + d) / (a + b + c + d) for a, b, c, d in rows),
            Fraction(),
        )
        s = sum(
            (c * (a + b) / (a + b + c + d) for a, b, c, d in rows),
            Fraction(),
        )
        # This positive identity is algebraically equivalent to the usual
        # subtraction form and gives the exact reference without cancellation.
        p = sum(
            (
                (a * d * (a + b) + b * c * (c + d)) / (a + b + c + d) ** 2
                for a, b, c, d in rows
            ),
            Fraction(),
        )
        variance = p / (r * s)
        weights = [c * (a + b) / (a + b + c + d) for a, b, c, d in rows]
        return FormulaFit(
            _log_fraction(r / s),
            math.sqrt(float(variance)),
            _normalize(weights),
        )

    if measure != "RD":
        raise ValueError(f"Unsupported measure: {measure}")
    weights = [(a + b) * (c + d) / (a + b + c + d) for a, b, c, d in rows]
    weight_sum = sum(weights, Fraction())
    estimate = (
        sum(
            (
                weight * (a / (a + b) - c / (c + d))
                for weight, (a, b, c, d) in zip(weights, rows, strict=True)
            ),
            Fraction(),
        )
        / weight_sum
    )
    linear = sum(
        (
            c * ((a + b) / (a + b + c + d)) ** 2
            - a * ((c + d) / (a + b + c + d)) ** 2
            + (a + b) * (c + d) * ((c + d) - (a + b)) / (2 * (a + b + c + d) ** 2)
            for a, b, c, d in rows
        ),
        Fraction(),
    )
    binomial = sum(
        (
            (a * ((c + d) - c) / (a + b + c + d) + c * ((a + b) - a) / (a + b + c + d))
            / 2
            for a, b, c, d in rows
        ),
        Fraction(),
    )
    variance = (estimate * linear + binomial) / weight_sum**2
    return FormulaFit(
        float(estimate),
        math.sqrt(float(variance)),
        _normalize(weights),
    )


def _peto_formula(
    a_values: np.ndarray,
    b_values: np.ndarray,
    c_values: np.ndarray,
    d_values: np.ndarray,
) -> FormulaFit:
    rows = [
        tuple(_fraction(value) for value in row)
        for row in zip(a_values, b_values, c_values, d_values, strict=True)
    ]
    observed_minus_expected: list[Fraction] = []
    information: list[Fraction] = []
    for a, b, c, d in rows:
        n1 = a + b
        n0 = c + d
        total = n1 + n0
        observed_minus_expected.append((a * n0 - c * n1) / total)
        information.append((a + c) * (b + d) * n1 * n0 / (total**2 * (total - 1)))
    information_sum = sum(information, Fraction())
    estimate = sum(observed_minus_expected, Fraction()) / information_sum
    q = sum(
        (
            (oe - estimate * variance_weight) ** 2 / variance_weight
            for oe, variance_weight in zip(
                observed_minus_expected, information, strict=True
            )
        ),
        Fraction(),
    )
    return FormulaFit(
        float(estimate),
        math.sqrt(float(1 / information_sum)),
        _normalize(information),
        float(q),
    )


def _fit_mh_without_runtime_warnings(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    measure: str,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return fit_mantel_haenszel(
            a,
            b,
            c,
            d,
            measure=measure,
            confidence_level=0.95,
        )


@pytest.mark.parametrize("measure", ["OR", "RR", "RD"])
def test_mh_matches_exact_fraction_formula_across_extreme_stratum_scales(
    measure: str,
) -> None:
    a = np.asarray([1.0, 5e307])
    b = np.asarray([9.0, 5e307])
    c = np.asarray([2.0, 5e307])
    d = np.asarray([8.0, 5e307])

    actual = _fit_mh_without_runtime_warnings(a, b, c, d, measure)
    expected = _mh_formula(a, b, c, d, measure)

    assert actual.estimate == pytest.approx(expected.estimate, rel=2e-13, abs=5e-15)
    assert actual.standard_error == pytest.approx(
        expected.standard_error, rel=2e-13, abs=5e-324
    )
    np.testing.assert_allclose(
        actual.normalized_weights,
        expected.normalized_weights,
        rtol=2e-13,
        atol=5e-324,
    )


def test_mh_or_retains_finite_opposing_cross_products() -> None:
    a = np.asarray([1.0])
    b = np.asarray([1e308])
    c = np.asarray([1e308])
    d = np.asarray([1.0])

    actual = _fit_mh_without_runtime_warnings(a, b, c, d, "OR")
    expected = _mh_formula(a, b, c, d, "OR")

    assert actual.estimate == pytest.approx(expected.estimate, rel=2e-15)
    assert actual.standard_error == pytest.approx(expected.standard_error, rel=2e-15)
    np.testing.assert_array_equal(actual.normalized_weights, [1.0])


def test_peto_matches_exact_fraction_formula_across_extreme_stratum_scales() -> None:
    a = np.asarray([1.0, 5e307])
    b = np.asarray([9.0, 5e307])
    c = np.asarray([2.0, 5e307])
    d = np.asarray([8.0, 5e307])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = fit_peto(a, b, c, d, confidence_level=0.95)
    expected = _peto_formula(a, b, c, d)

    assert actual.estimate == pytest.approx(expected.estimate, rel=2e-13, abs=5e-324)
    assert actual.standard_error == pytest.approx(
        expected.standard_error, rel=2e-13, abs=5e-324
    )
    assert actual.q == pytest.approx(expected.q, rel=2e-13, abs=5e-15)
    np.testing.assert_allclose(
        actual.normalized_weights,
        expected.normalized_weights,
        rtol=2e-13,
        atol=5e-324,
    )


def test_peto_signed_log_sum_preserves_exact_zero_effect() -> None:
    a = np.asarray([1.0, 2.0])
    b = np.asarray([9.0, 8.0])
    c = np.asarray([2.0, 1.0])
    d = np.asarray([8.0, 9.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = fit_peto(a, b, c, d, confidence_level=0.95)
    expected = _peto_formula(a, b, c, d)

    assert actual.estimate == 0.0
    assert expected.estimate == 0.0
    assert actual.standard_error == pytest.approx(expected.standard_error, rel=2e-15)
    assert actual.q == pytest.approx(expected.q, rel=2e-15)


def test_peto_rejects_zero_arms_and_too_small_strata_without_runtime_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ma.InvalidStudyDataError, match="positive group totals.*0"):
            fit_peto(
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 9.0]),
                np.asarray([1.0, 2.0]),
                np.asarray([9.0, 8.0]),
                confidence_level=0.95,
            )
        with pytest.raises(
            ma.InvalidStudyDataError, match="total sample size above one"
        ):
            fit_peto(
                np.asarray([0.25]),
                np.asarray([0.25]),
                np.asarray([0.25]),
                np.asarray([0.25]),
                confidence_level=0.95,
            )


@pytest.mark.parametrize("method", ["MH", "Peto"])
def test_public_sparse_estimators_accept_extreme_mixed_stratum_scales(
    method: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = ma.meta_binary(
            event_treat=[1.0, 5e307],
            n_treat=[10.0, 1e308],
            event_control=[2.0, 5e307],
            n_control=[10.0, 1e308],
            measure="RR" if method == "MH" else "OR",
            method=method,
        )

    assert np.isfinite(result.estimate)
    assert np.isfinite(result.standard_error)
    assert result.standard_error > 0.0
    assert result.study_results["included"].all()
    assert result.study_results["normalized_weight"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("measure", ["OR", "RR", "RD"])
def test_mh_extreme_scale_results_are_row_permutation_invariant(measure: str) -> None:
    a = np.asarray([3.0, 2e250, 7e100])
    b = np.asarray([17.0, 8e250, 4e100])
    c = np.asarray([5.0, 4e250, 2e100])
    d = np.asarray([19.0, 6e250, 9e100])
    order = np.asarray([2, 0, 1])

    original = _fit_mh_without_runtime_warnings(a, b, c, d, measure)
    permuted = _fit_mh_without_runtime_warnings(
        a[order], b[order], c[order], d[order], measure
    )

    assert permuted.estimate == pytest.approx(original.estimate, rel=2e-13, abs=5e-15)
    assert permuted.standard_error == pytest.approx(
        original.standard_error, rel=2e-13, abs=5e-324
    )
    np.testing.assert_allclose(
        permuted.normalized_weights[np.argsort(order)],
        original.normalized_weights,
        rtol=2e-13,
        atol=5e-324,
    )


@pytest.mark.parametrize("measure", ["OR", "RR"])
def test_mh_relative_effects_respect_common_count_scaling(measure: str) -> None:
    a = np.asarray([3.0, 8.0, 14.0, 5.0])
    b = np.asarray([47.0, 72.0, 86.0, 55.0])
    c = np.asarray([5.0, 6.0, 18.0, 9.0])
    d = np.asarray([45.0, 74.0, 82.0, 51.0])
    scale = 1e250

    ordinary = _fit_mh_without_runtime_warnings(a, b, c, d, measure)
    scaled = _fit_mh_without_runtime_warnings(
        a * scale, b * scale, c * scale, d * scale, measure
    )

    assert scaled.estimate == pytest.approx(ordinary.estimate, rel=2e-13, abs=5e-15)
    assert scaled.standard_error == pytest.approx(
        ordinary.standard_error / math.sqrt(scale), rel=2e-13
    )
    np.testing.assert_allclose(
        scaled.normalized_weights,
        ordinary.normalized_weights,
        rtol=2e-13,
        atol=5e-15,
    )


def test_peto_extreme_scale_results_are_symmetric_under_arm_swap() -> None:
    a = np.asarray([3.0, 2e250, 7e100])
    b = np.asarray([17.0, 8e250, 4e100])
    c = np.asarray([5.0, 4e250, 2e100])
    d = np.asarray([19.0, 6e250, 9e100])

    original = fit_peto(a, b, c, d, confidence_level=0.95)
    swapped = fit_peto(c, d, a, b, confidence_level=0.95)

    assert swapped.estimate == pytest.approx(-original.estimate, rel=2e-13, abs=5e-15)
    assert swapped.standard_error == pytest.approx(original.standard_error, rel=2e-13)
    assert swapped.q == pytest.approx(original.q, rel=2e-13, abs=5e-15)
    np.testing.assert_allclose(
        swapped.normalized_weights,
        original.normalized_weights,
        rtol=2e-13,
        atol=5e-324,
    )


def _case_arguments(case: dict[str, Any]) -> dict[str, list[float]]:
    return {
        name: [float(value) for value in values]
        for name, values in case["input"].items()
    }


def _assert_public_fit(
    result: ma.MetaAnalysisResult,
    expected: dict[str, Any],
    *,
    compare_heterogeneity: bool,
) -> None:
    np.testing.assert_allclose(
        [result.estimate, result.standard_error, result.ci_low, result.ci_high],
        [expected["estimate"], expected["standard_error"], *expected["ci"]],
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )
    included = result.study_results.loc[lambda frame: frame["included"]]
    np.testing.assert_allclose(
        included["normalized_weight"],
        expected["weights"],
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )
    if compare_heterogeneity:
        heterogeneity = expected["heterogeneity"]
        np.testing.assert_allclose(
            [result.q, result.q_pvalue, result.i2, result.h2],
            [
                heterogeneity["q"],
                heterogeneity["pvalue"],
                heterogeneity["i2"],
                heterogeneity["h2"],
            ],
            rtol=REFERENCE_RTOL,
            atol=REFERENCE_ATOL,
        )
        assert result.q_df == heterogeneity["df"]


def test_sparse_binary_review_fixture_records_pinned_environment() -> None:
    assert R_REFERENCE["generated_by"] == "R metafor"
    assert R_REFERENCE["r_version"] == "R version 4.6.1 (2026-06-24 ucrt)"
    assert R_REFERENCE["metafor_version"] == "5.0.1"
    assert R_REFERENCE["jsonlite_version"] == "2.0.0"
    assert R_REFERENCE["confidence_level"] == pytest.approx(0.95)
    assert R_REFERENCE["study_correction"] == {"add": 0.5, "scope": "only0"}
    assert R_REFERENCE["mh_pooling_corrections"] == {
        "raw": 0,
        "only_zero": 0.5,
    }


@pytest.mark.parametrize("case_name", list(R_REFERENCE["cases"]))
@pytest.mark.parametrize("measure", ["OR", "RR", "RD"])
def test_sparse_binary_study_effects_match_metafor_case_matrix(
    case_name: str,
    measure: str,
) -> None:
    case = R_REFERENCE["cases"][case_name]
    arguments = _case_arguments(case)
    expected = case["measures"][measure]["studies"]
    result = ma.meta_binary(**arguments, measure=measure, method="MH")
    studies = result.study_results
    expected_effect = np.asarray(expected["effect"], dtype=np.float64)
    expected_variance = np.asarray(expected["variance"], dtype=np.float64)

    np.testing.assert_allclose(
        studies["variance"],
        expected_variance,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        equal_nan=True,
    )
    if measure == "RD":
        raw_effect = np.asarray(arguments["event_treat"]) / np.asarray(
            arguments["n_treat"]
        ) - np.asarray(arguments["event_control"]) / np.asarray(arguments["n_control"])
        np.testing.assert_allclose(
            studies["effect"], raw_effect, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
        )
        assert studies["included"].all()
    else:
        np.testing.assert_allclose(
            studies["effect"],
            expected_effect,
            rtol=REFERENCE_RTOL,
            atol=REFERENCE_ATOL,
            equal_nan=True,
        )
        assert studies["included"].tolist() == np.isfinite(expected_effect).tolist()


@pytest.mark.parametrize("case_name", list(R_REFERENCE["cases"]))
@pytest.mark.parametrize("measure", ["OR", "RR", "RD"])
def test_sparse_binary_mh_fits_match_metafor_case_matrix(
    case_name: str,
    measure: str,
) -> None:
    case = R_REFERENCE["cases"][case_name]
    arguments = _case_arguments(case)
    expected = case["measures"][measure]
    cells = np.column_stack(
        [
            arguments["event_treat"],
            np.asarray(arguments["n_treat"]) - arguments["event_treat"],
            arguments["event_control"],
            np.asarray(arguments["n_control"]) - arguments["event_control"],
        ]
    )

    raw = ma.meta_binary(**arguments, measure=measure, method="MH")
    corrected = ma.meta_binary(
        **arguments,
        measure=measure,
        method="MH",
        mh_continuity_correction=0.5,
    )

    compare_r_heterogeneity = measure != "RD" or not np.any(cells == 0.0)
    _assert_public_fit(
        raw,
        expected["mh_raw"],
        compare_heterogeneity=compare_r_heterogeneity,
    )
    _assert_public_fit(corrected, expected["mh_only_zero"], compare_heterogeneity=False)
    if measure == "RD" and not compare_r_heterogeneity:
        studies = raw.study_results.loc[lambda frame: frame["included"]]
        inverse_variance = 1.0 / studies["variance"].to_numpy()
        expected_q = float(
            np.sum(inverse_variance * (studies["effect"] - raw.estimate) ** 2)
        )
        assert raw.q == pytest.approx(expected_q, rel=REFERENCE_RTOL)
        assert raw.q_df == len(studies) - 1
        expected_i2 = (
            0.0 if expected_q <= 0 else max(0.0, (expected_q - raw.q_df) / expected_q)
        )
        assert raw.i2 == pytest.approx(expected_i2, rel=REFERENCE_RTOL)
        assert raw.h2 == pytest.approx(expected_q / raw.q_df, rel=REFERENCE_RTOL)
    expected_corrected = corrected.study_results["included"].to_numpy() & np.any(
        cells == 0.0, axis=1
    )
    np.testing.assert_array_equal(
        corrected.study_results["mh_continuity_corrected"], expected_corrected
    )


@pytest.mark.parametrize("case_name", list(R_REFERENCE["cases"]))
def test_sparse_binary_peto_fits_match_metafor_case_matrix(case_name: str) -> None:
    case = R_REFERENCE["cases"][case_name]
    arguments = _case_arguments(case)
    expected = case["peto"]
    result = ma.meta_binary(**arguments, measure="OR", method="Peto")
    studies = result.study_results

    np.testing.assert_allclose(
        studies["effect"],
        np.asarray(expected["effect"], dtype=np.float64),
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        studies["variance"],
        np.asarray(expected["variance"], dtype=np.float64),
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        equal_nan=True,
    )
    _assert_public_fit(result, expected["fit"], compare_heterogeneity=True)
