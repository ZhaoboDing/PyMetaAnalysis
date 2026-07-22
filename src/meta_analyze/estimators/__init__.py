"""Numerical estimators used by the public meta-analysis API."""

from .inverse_variance import InverseVarianceFit, fit_inverse_variance
from .mantel_haenszel import MantelHaenszelFit, fit_mantel_haenszel
from .meta_regression import (
    MetaRegressionFit,
    RegressionTestFit,
    estimate_meta_regression_tau2,
    fit_meta_regression,
    residual_heterogeneity,
)
from .tau2 import Tau2Estimate, estimate_tau2

__all__ = [
    "InverseVarianceFit",
    "MantelHaenszelFit",
    "MetaRegressionFit",
    "RegressionTestFit",
    "Tau2Estimate",
    "estimate_tau2",
    "estimate_meta_regression_tau2",
    "fit_inverse_variance",
    "fit_mantel_haenszel",
    "fit_meta_regression",
    "residual_heterogeneity",
]
