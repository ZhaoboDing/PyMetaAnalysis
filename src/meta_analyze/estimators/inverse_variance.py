"""Common- and random-effects inverse-variance estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, t

from ..exceptions import InsufficientStudiesError, UnsupportedMethodError
from ..heterogeneity import weighted_mean
from .tau2 import Tau2Estimate, estimate_tau2


@dataclass(frozen=True, slots=True)
class InverseVarianceFit:
    """Numerical outputs from an inverse-variance model."""

    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    prediction_interval: tuple[float, float] | None
    weights: NDArray[np.float64]
    normalized_weights: NDArray[np.float64]
    tau2: Tau2Estimate | None
    warnings: tuple[str, ...]


def _confidence_interval(
    *,
    effect: NDArray[np.float64],
    weights: NDArray[np.float64],
    estimate: float,
    classic_variance: float,
    ci_method: str,
    confidence_level: float,
) -> tuple[float, float, float, tuple[str, ...]]:
    alpha = 1.0 - confidence_level
    warnings: list[str] = []

    if ci_method == "normal":
        variance = classic_variance
        critical_value = float(norm.ppf(1.0 - alpha / 2.0))
    else:
        df = len(effect) - 1
        if bool(np.all(effect == effect[0])):
            # Weighted means can differ from an identical input value by one ULP,
            # depending on the platform's floating-point reduction. The residual
            # variation is mathematically zero in this case, so preserve that
            # invariant explicitly instead of squaring the rounding error.
            scale = 0.0
        else:
            residual = effect - estimate
            scale = float(np.dot(weights, residual * residual) / df)
        hk_variance = scale / float(np.sum(weights))
        if ci_method == "hartung_knapp_adhoc":
            variance = max(classic_variance, hk_variance)
        else:
            variance = hk_variance
            if hk_variance < classic_variance:
                warnings.append(
                    "Hartung-Knapp produced a variance below the classic variance; "
                    "use ci_method='hartung_knapp_adhoc' for lower-bound protection."
                )
        if variance == 0.0:
            warnings.append(
                "Hartung-Knapp variance is zero because the included effects have "
                "no weighted residual variation."
            )
        critical_value = float(t.ppf(1.0 - alpha / 2.0, df=df))

    standard_error = float(np.sqrt(variance))
    margin = critical_value * standard_error
    return estimate - margin, estimate + margin, standard_error, tuple(warnings)


def fit_inverse_variance(
    effect: NDArray[np.float64],
    variance: NDArray[np.float64],
    *,
    model: str,
    tau2_method: str,
    ci_method: str,
    confidence_level: float,
    atol: float,
    max_iter: int,
) -> InverseVarianceFit:
    """Fit a common- or random-effects inverse-variance model."""

    if model == "random" and len(effect) < 2:
        raise InsufficientStudiesError(
            "A random-effects model requires at least two included studies."
        )
    if ci_method not in {"normal", "hartung_knapp", "hartung_knapp_adhoc"}:
        raise UnsupportedMethodError(
            "ci_method must be 'normal', 'hartung_knapp', or 'hartung_knapp_adhoc'."
        )
    if model == "common" and ci_method != "normal":
        raise UnsupportedMethodError(
            "Hartung-Knapp intervals are only supported for random-effects models."
        )

    tau2: Tau2Estimate | None
    if model == "common":
        tau2 = None
        tau2_value = 0.0
    else:
        tau2 = estimate_tau2(
            effect,
            variance,
            method=tau2_method,
            atol=atol,
            max_iter=max_iter,
        )
        tau2_value = tau2.value

    denominator = variance + tau2_value
    weights = 1.0 / denominator
    variance_scale = float(np.min(denominator))
    relative_weights = variance_scale / denominator
    relative_weight_sum = float(np.sum(relative_weights))
    normalized_weights = relative_weights / relative_weight_sum
    estimate = weighted_mean(effect, relative_weights)
    classic_variance = variance_scale / relative_weight_sum
    ci_low, ci_high, standard_error, interval_warnings = _confidence_interval(
        effect=effect,
        weights=normalized_weights,
        estimate=estimate,
        classic_variance=classic_variance,
        ci_method=ci_method,
        confidence_level=confidence_level,
    )

    warnings = list(interval_warnings)
    prediction_interval: tuple[float, float] | None = None
    if model == "random":
        if len(effect) >= 3:
            prediction_df = len(effect) - 2
            critical_value = float(
                t.ppf(0.5 + confidence_level / 2.0, df=prediction_df)
            )
            # HTS uses the classic variance of the pooled mean. This remains
            # independent of any Hartung-Knapp method used for the mean CI.
            prediction_se = float(np.sqrt(tau2_value + classic_variance))
            margin = critical_value * prediction_se
            prediction_interval = (estimate - margin, estimate + margin)
            if len(effect) < 5:
                warnings.append(
                    "Prediction intervals are especially uncertain with fewer "
                    "than five included studies."
                )
        else:
            warnings.append(
                "A prediction interval requires at least three included studies."
            )

    return InverseVarianceFit(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        prediction_interval=prediction_interval,
        weights=weights,
        normalized_weights=normalized_weights,
        tau2=tau2,
        warnings=tuple(warnings),
    )
