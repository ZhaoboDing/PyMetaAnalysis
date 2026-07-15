"""Pandas-first tools for auditable meta-analysis workflows."""

from .api import meta_analysis
from .binary_api import meta_binary
from .config import MethodConfig
from .continuous_api import meta_continuous
from .exceptions import (
    ConvergenceError,
    InsufficientStudiesError,
    InvalidStudyDataError,
    MetaAnalysisError,
    UnsupportedMethodError,
)
from .results import (
    FitDiagnostics,
    HeterogeneityResult,
    MetaAnalysisResult,
    MetaAnalysisSummary,
)

__all__ = [
    "ConvergenceError",
    "FitDiagnostics",
    "HeterogeneityResult",
    "InsufficientStudiesError",
    "InvalidStudyDataError",
    "MetaAnalysisError",
    "MetaAnalysisResult",
    "MetaAnalysisSummary",
    "MethodConfig",
    "UnsupportedMethodError",
    "meta_analysis",
    "meta_binary",
    "meta_continuous",
]

__version__ = "0.1.0.dev0"
