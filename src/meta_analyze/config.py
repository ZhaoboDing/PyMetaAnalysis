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
    options: tuple[tuple[str, MethodOptionValue], ...]
