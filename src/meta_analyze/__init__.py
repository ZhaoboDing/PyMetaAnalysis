"""Pandas-first tools for auditable meta-analysis workflows."""

from ._version import __version__
from .api import meta_analysis
from .binary_api import meta_binary
from .config import MethodConfig, SubgroupMethodConfig
from .continuous_api import meta_continuous
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
    "FitDiagnostics",
    "HeterogeneityResult",
    "InsufficientStudiesError",
    "InputFieldProvenance",
    "InvalidStudyDataError",
    "LeaveOneOutResult",
    "MetaAnalysisError",
    "MetaAnalysisResult",
    "MetaAnalysisSummary",
    "MethodConfig",
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
    "__version__",
]
