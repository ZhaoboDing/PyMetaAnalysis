"""Pandas-first tools for auditable meta-analysis workflows."""

from ._version import __version__
from .api import meta_analysis
from .binary_api import meta_binary
from .config import MetaRegressionMethodConfig, MethodConfig, SubgroupMethodConfig
from .continuous_api import meta_continuous
from .design_matrix import DesignInfo, ModeratorSpec
from .exceptions import (
    ConvergenceError,
    InsufficientStudiesError,
    InvalidStudyDataError,
    MetaAnalysisError,
    UnsupportedMethodError,
)
from .provenance import (
    AnalysisProvenance,
    InputFieldProvenance,
    TransformationRecord,
)
from .regression_api import meta_regression
from .regression_results import (
    MetaRegressionDiagnostics,
    MetaRegressionResult,
    MetaRegressionSummary,
    ModeratorTestResult,
)
from .reporting import ResultReport
from .results import (
    FitDiagnostics,
    HeterogeneityResult,
    MetaAnalysisResult,
    MetaAnalysisSummary,
    SubgroupMetaAnalysisResult,
    SubgroupMetaAnalysisSummary,
)
from .sensitivity import (
    CumulativeMetaAnalysisResult,
    LeaveOneOutResult,
    SubgroupCumulativeMetaAnalysisResult,
    SubgroupLeaveOneOutResult,
)

__all__ = [
    "AnalysisProvenance",
    "ConvergenceError",
    "CumulativeMetaAnalysisResult",
    "DesignInfo",
    "FitDiagnostics",
    "HeterogeneityResult",
    "InsufficientStudiesError",
    "InputFieldProvenance",
    "InvalidStudyDataError",
    "LeaveOneOutResult",
    "MetaAnalysisError",
    "MetaAnalysisResult",
    "MetaAnalysisSummary",
    "MetaRegressionDiagnostics",
    "MetaRegressionMethodConfig",
    "MetaRegressionResult",
    "MetaRegressionSummary",
    "MethodConfig",
    "ModeratorSpec",
    "ModeratorTestResult",
    "ResultReport",
    "SubgroupMetaAnalysisResult",
    "SubgroupMetaAnalysisSummary",
    "SubgroupCumulativeMetaAnalysisResult",
    "SubgroupLeaveOneOutResult",
    "SubgroupMethodConfig",
    "TransformationRecord",
    "UnsupportedMethodError",
    "meta_analysis",
    "meta_binary",
    "meta_continuous",
    "meta_regression",
    "__version__",
]
