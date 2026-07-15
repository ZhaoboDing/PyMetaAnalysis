"""Domain-specific exceptions raised by :mod:`meta_analyze`."""


class MetaAnalysisError(Exception):
    """Base class for all library-specific errors."""


class InvalidStudyDataError(MetaAnalysisError, ValueError):
    """Raised when study data cannot be interpreted safely."""


class InsufficientStudiesError(MetaAnalysisError, ValueError):
    """Raised when a requested method needs more included studies."""


class ConvergenceError(MetaAnalysisError, RuntimeError):
    """Raised when an iterative estimator fails to converge."""


class UnsupportedMethodError(MetaAnalysisError, ValueError):
    """Raised when a model or estimator name is unsupported."""
