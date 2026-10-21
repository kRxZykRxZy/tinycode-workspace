"""Utilities shared across the inference engine."""

from __future__ import annotations

from .errors import (
    ArtifactError,
    BackendError,
    BudgetError,
    ConfigurationError,
    EngineError,
    ModelError,
    RequestError,
    SchedulingError,
)
from .platform import cpu_count, detect_architecture, detect_simd, is_arm, is_x86
from .timers import Stopwatch, Timer, now_ms
from .version import ContractVersion, engine_version

__all__ = [
    "ArtifactError",
    "BackendError",
    "BudgetError",
    "ConfigurationError",
    "ContractVersion",
    "EngineError",
    "ModelError",
    "RequestError",
    "SchedulingError",
    "Stopwatch",
    "Timer",
    "cpu_count",
    "detect_architecture",
    "detect_simd",
    "engine_version",
    "is_arm",
    "is_x86",
    "now_ms",
]
