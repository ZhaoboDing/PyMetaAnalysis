from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import meta_analyze as ma

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_DATA = pd.read_csv(REFERENCE_DIR / "meta_regression_input.csv")
REFERENCE: dict[str, Any] = json.loads(
    (REFERENCE_DIR / "meta_regression_contrasts_metafor.json").read_text(
        encoding="utf-8"
    )
)


def _reference_fit(**options: str) -> ma.MetaRegressionResult:
    return ma.meta_regression(
        REFERENCE_DATA,
        effect="effect",
        variance="variance",
        moderators=["mean_age", "dose", "region"],
        categorical={"region": REFERENCE["categorical_levels"]["region"]},
        study="study",
        atol=1e-10,
        max_iter=1000,
        **options,  # type: ignore[arg-type]
    )


def _reference_contrasts(
    result: ma.MetaRegressionResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        REFERENCE["contrast_matrix"],
        index=REFERENCE["contrast_names"],
        columns=result.design_info.term_names,
    )


@pytest.mark.parametrize(
    ("reference_name", "options"),
    [
        ("common", {"model": "common"}),
        ("mixed_reml", {"model": "mixed", "tau2_method": "REML"}),
        (
            "mixed_reml_hartung_knapp",
            {
                "model": "mixed",
                "tau2_method": "REML",
                "inference_method": "hartung_knapp",
            },
        ),
        (
            "mixed_reml_hartung_knapp_adhoc",
            {
                "model": "mixed",
                "tau2_method": "REML",
                "inference_method": "hartung_knapp_adhoc",
            },
        ),
    ],
)
def test_meta_regression_contrasts_match_metafor(
    reference_name: str, options: dict[str, str]
) -> None:
    result = _reference_fit(**options)
    contrasts = result.contrast(
        _reference_contrasts(result),
        rhs=dict(zip(REFERENCE["contrast_names"], REFERENCE["rhs"], strict=True)),
    )
    expected = REFERENCE["models"][reference_name]

    assert REFERENCE["metafor_version"] == "5.0.1"
    assert contrasts.table["contrast"].tolist() == REFERENCE["contrast_names"]
    for actual_column, expected_column in [
        ("estimate", "estimate"),
        ("standard_error", "standard_error"),
        ("statistic", "statistic"),
        ("pvalue", "pvalue"),
    ]:
        np.testing.assert_allclose(
            contrasts.table[actual_column],
            expected[expected_column],
            rtol=2e-8,
            atol=1e-10,
        )
    assert contrasts.joint_test.statistic == pytest.approx(
        expected["joint_statistic"], rel=2e-8, abs=1e-10
    )
    assert contrasts.joint_test.pvalue == pytest.approx(
        expected["joint_pvalue"], rel=2e-8, abs=1e-10
    )
    assert contrasts.joint_test.df_num == expected["joint_df"][0]
    assert contrasts.joint_test.df_denom == expected["joint_df"][1]


def test_single_nonzero_rhs_matches_manual_wald_calculation() -> None:
    result = ma.meta_regression(
        effect=[0.0, 0.3, 0.5, 0.9, 1.0, 1.4],
        variance=[0.04, 0.05, 0.06, 0.05, 0.07, 0.08],
        moderators={
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        },
        model="common",
    )
    weights = {"intercept": 0.5, "x": 2.0, "z": -1.0}
    rhs = 0.25

    contrast = result.contrast(weights, name="prespecified", rhs=rhs)
    vector = np.asarray([0.5, 2.0, -1.0])
    beta = result.coefficients["estimate"].to_numpy()
    covariance = result.coefficient_covariance.to_numpy()
    expected_estimate = float(vector @ beta)
    expected_se = float(np.sqrt(vector @ covariance @ vector))
    expected_z = (expected_estimate - rhs) / expected_se
    critical = float(norm.ppf(0.975))
    row = contrast.table.iloc[0]

    assert row["contrast"] == "prespecified"
    assert row["estimate"] == pytest.approx(expected_estimate)
    assert row["rhs"] == rhs
    assert row["estimate_minus_rhs"] == pytest.approx(expected_estimate - rhs)
    assert row["standard_error"] == pytest.approx(expected_se)
    assert row["statistic"] == pytest.approx(expected_z)
    assert row["ci_low"] == pytest.approx(expected_estimate - critical * expected_se)
    assert row["ci_high"] == pytest.approx(expected_estimate + critical * expected_se)
    assert contrast.joint_test.statistic == pytest.approx(expected_z**2)
    assert contrast.joint_test.pvalue == pytest.approx(row["pvalue"])
    assert contrast.warnings == ()


def test_named_mapping_and_dataframe_fill_unspecified_terms_with_zero() -> None:
    result = _reference_fit(model="common")
    named = result.contrast(
        {
            "age": {"mean_age": 1.0},
            "regional_difference": {
                "region[South]": 1.0,
                "region[East]": -1.0,
            },
        },
        rhs={"age": 0.01, "regional_difference": 0.0},
    )
    framed = result.contrast(
        pd.DataFrame(
            {
                "mean_age": [1.0, 0.0],
                "region[South]": [0.0, 1.0],
                "region[East]": [0.0, -1.0],
            },
            index=["age", "regional_difference"],
        ),
        rhs={"age": 0.01, "regional_difference": 0.0},
    )

    pd.testing.assert_frame_equal(named.table, framed.table)
    pd.testing.assert_frame_equal(named.contrast_matrix, framed.contrast_matrix)
    assert named.contrast_matrix["intercept"].eq(0.0).all()
    assert named.contrast_matrix["dose"].eq(0.0).all()
    assert len(named) == 2
    assert named.pvalue_adjustment == "none"
    assert len(named.warnings) == 1
    assert "unadjusted" in named.warnings[0]


def test_hartung_knapp_contrasts_use_t_and_joint_f_inference() -> None:
    result = _reference_fit(
        model="mixed",
        tau2_method="REML",
        inference_method="hartung_knapp",
    )

    contrasts = result.contrast(_reference_contrasts(result))

    assert contrasts.table["statistic_name"].eq("t").all()
    assert contrasts.table["distribution"].eq("t").all()
    assert contrasts.table["df"].eq(float(result.residual_df)).all()
    assert contrasts.joint_test.statistic_name == "F"
    assert contrasts.joint_test.distribution == "F"
    assert contrasts.joint_test.df_num == len(contrasts)
    assert contrasts.joint_test.df_denom == result.residual_df


def test_scientifically_equivalent_contrasts_survive_reparameterization() -> None:
    north = _reference_fit(model="common")
    south = ma.meta_regression(
        REFERENCE_DATA,
        effect="effect",
        variance="variance",
        moderators=["mean_age", "dose", "region"],
        categorical={"region": ["South", "North", "East"]},
        study="study",
        model="common",
    )
    north_difference = north.contrast({"region[South]": 1.0, "region[East]": -1.0})
    south_difference = south.contrast({"region[East]": -1.0})

    for column in ["estimate", "standard_error", "statistic", "pvalue"]:
        assert north_difference.table.loc[0, column] == pytest.approx(
            south_difference.table.loc[0, column], rel=2e-10, abs=2e-12
        )

    rescaled_data = REFERENCE_DATA.copy()
    rescaled_data["mean_age"] *= 100.0
    rescaled = ma.meta_regression(
        rescaled_data,
        effect="effect",
        variance="variance",
        moderators=["mean_age", "dose"],
        study="study",
        model="common",
    )
    original = ma.meta_regression(
        REFERENCE_DATA,
        effect="effect",
        variance="variance",
        moderators=["mean_age", "dose"],
        study="study",
        model="common",
    )
    original_age = original.contrast({"mean_age": 1.0})
    rescaled_age = rescaled.contrast({"mean_age": 100.0})
    np.testing.assert_allclose(
        original_age.table[["estimate", "standard_error", "statistic", "pvalue"]],
        rescaled_age.table[["estimate", "standard_error", "statistic", "pvalue"]],
        rtol=2e-10,
        atol=2e-12,
    )


def test_no_intercept_contrast_is_supported() -> None:
    result = ma.meta_regression(
        effect=[0.1, 0.4, 0.8, 1.1, 1.4],
        variance=[0.04, 0.05, 0.06, 0.07, 0.08],
        moderators={
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.0, 1.0, 0.0, 1.0, 0.0],
        },
        model="common",
        intercept=False,
    )

    contrast = result.contrast({"x": 1.0, "z": -1.0}, name="x_minus_z")

    assert contrast.contrast_matrix.columns.tolist() == ["x", "z"]
    assert contrast.table.loc[0, "contrast"] == "x_minus_z"


@pytest.mark.parametrize(
    ("contrasts", "match"),
    [
        ({}, "non-empty"),
        ({"unknown": 1.0}, "unknown terms"),
        ({"mean_age": 0.0}, "nonzero"),
        ({"mean_age": True}, "real number"),
        ({"mean_age": np.inf}, "finite"),
        (
            {
                "first": {"mean_age": 1.0},
                "second": {"mean_age": 2.0},
            },
            "full row rank",
        ),
        (
            {
                "first": {"mean_age": 1.0},
                "second": 2.0,
            },
            "either only real",
        ),
    ],
)
def test_invalid_contrast_mappings_raise_domain_errors(
    contrasts: Any, match: str
) -> None:
    result = _reference_fit(model="common")

    with pytest.raises(ma.InvalidStudyDataError, match=match):
        result.contrast(contrasts)


def test_invalid_contrast_dataframe_and_rhs_raise_domain_errors() -> None:
    result = _reference_fit(model="common")
    duplicate_names = pd.DataFrame(
        {"mean_age": [1.0, 2.0]},
        index=["duplicate", "duplicate"],
    )
    duplicate_terms = pd.DataFrame(
        [[1.0, 2.0]],
        index=["duplicate_terms"],
        columns=["mean_age", "mean_age"],
    )

    with pytest.raises(ma.InvalidStudyDataError, match="index names must be unique"):
        result.contrast(duplicate_names)
    with pytest.raises(ma.InvalidStudyDataError, match="term columns must be unique"):
        result.contrast(duplicate_terms)
    with pytest.raises(ma.InvalidStudyDataError, match="non-empty strings"):
        result.contrast(pd.DataFrame({"mean_age": [1.0]}, index=[0]))
    with pytest.raises(ma.InvalidStudyDataError, match="only supported"):
        result.contrast({"age": {"mean_age": 1.0}}, name="override")
    with pytest.raises(ma.InvalidStudyDataError, match="exactly"):
        result.contrast(
            {"age": {"mean_age": 1.0}, "dose": {"dose": 1.0}},
            rhs={"age": 0.0},
        )
    with pytest.raises(ma.InvalidStudyDataError, match="real numbers"):
        result.contrast({"mean_age": 1.0}, rhs="zero")  # type: ignore[arg-type]
    with pytest.raises(ma.InvalidStudyDataError, match="finite"):
        result.contrast({"mean_age": 1.0}, rhs=np.nan)


def test_contrast_outputs_are_defensive_and_frozen() -> None:
    result = _reference_fit(model="common")
    contrasts = result.contrast(
        {
            "age": {"mean_age": 1.0},
            "dose": {"dose": 1.0},
        }
    )

    assert isinstance(contrasts, ma.MetaRegressionContrastResult)
    assert isinstance(contrasts.joint_test, ma.LinearContrastTestResult)
    pd.testing.assert_frame_equal(contrasts.summary(), contrasts.table)
    pd.testing.assert_frame_equal(contrasts.to_dataframe(), contrasts.table)
    table = contrasts.table
    table.loc[0, "estimate"] = 999.0
    assert contrasts.table.loc[0, "estimate"] != 999.0
    matrix = contrasts.contrast_matrix
    matrix.loc["age", "mean_age"] = 999.0
    assert contrasts.contrast_matrix.loc["age", "mean_age"] != 999.0
    payload = contrasts.joint_test.to_dict()
    assert payload["contrasts"] == ["age", "dose"]
    with pytest.raises(FrozenInstanceError):
        contrasts.pvalue_adjustment = "bonferroni"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contrasts.joint_test.pvalue = 1.0  # type: ignore[misc]
