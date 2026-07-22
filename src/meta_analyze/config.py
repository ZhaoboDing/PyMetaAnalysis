"""Configuration objects recorded in fitted results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

MethodOptionValue: TypeAlias = str | float | int | bool | None


@dataclass(frozen=True, slots=True)
class MethodConfig:
    """The fully resolved methods used to fit a meta-analysis."""

    model: str
    pooling_method: str
    tau2_method: str | None
    ci_method: str
    confidence_level: float
    prediction_interval_method: str | None
    missing: str
    atol: float
    max_iter: int
    options: tuple[tuple[str, MethodOptionValue], ...]


@dataclass(frozen=True, slots=True)
class SubgroupMethodConfig:
    """The fully resolved assumptions used for a subgroup analysis."""

    model: str
    tau2_strategy: str
    test_method: str
    subgroup_missing: str


@dataclass(frozen=True, slots=True)
class MetaRegressionMethodConfig:
    """The fully resolved assumptions used for a meta-regression."""

    model: str
    tau2_method: str | None
    inference_method: str
    confidence_level: float
    intercept: bool
    moderator_names: tuple[str, ...]
    term_names: tuple[str, ...]
    categorical_references: tuple[tuple[str, str], ...]
    prediction_interval_method: str | None
    missing: str
    atol: float
    max_iter: int
