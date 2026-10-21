"""Feature extraction for the adaptive budget controller.

The feature vector and its ordering form part of the shared contract
(``budget_features.v1``).  The inference engine implements the same
extraction so that a controller trained here can be loaded there.

Feature order (never reorder; append new features at the end and bump the
schema version):

0  prompt_tokens_normalised
1  requested_output_normalised
2  task_category_code
3  task_category_explain
4  task_category_transform
5  task_category_debug
6  task_category_structured
7  complexity_score
8  uncertainty_score
9  wants_verification
10 has_latency_target
11 latency_target_normalised
12 queue_pressure
13 memory_pressure
14 backend_speed
15 plan_steps_normalised
16 computation_budget_normalised
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "TASK_CATEGORIES",
    "BudgetFeatures",
    "feature_schema",
    "feature_vector",
]

FEATURE_SCHEMA_VERSION = "budget_features.v1"

FEATURE_NAMES: tuple[str, ...] = (
    "prompt_tokens_normalised",
    "requested_output_normalised",
    "task_category_code",
    "task_category_explain",
    "task_category_transform",
    "task_category_debug",
    "task_category_structured",
    "complexity_score",
    "uncertainty_score",
    "wants_verification",
    "has_latency_target",
    "latency_target_normalised",
    "queue_pressure",
    "memory_pressure",
    "backend_speed",
    "plan_steps_normalised",
    "computation_budget_normalised",
)

TASK_CATEGORIES: tuple[str, ...] = (
    "code",
    "explain",
    "transform",
    "debug",
    "structured",
)


def feature_schema() -> dict[str, Any]:
    """Return the machine readable feature schema."""

    return {
        "version": FEATURE_SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "categories": list(TASK_CATEGORIES),
        "normalisation": {
            "prompt_tokens": "min(prompt_tokens, 1024) / 1024",
            "requested_output": "min(requested_output_tokens, 1024) / 1024",
            "latency_target": "min(latency_target_ms, 120000) / 120000",
            "backend_speed": "min(tokens_per_second, 512) / 512",
            "plan_steps": "min(plan_steps, 8) / 8",
            "computation_budget": "min(computation_budget, 4096) / 4096",
        },
    }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


@dataclass
class BudgetFeatures:
    """Raw inputs that describe a request to the budget controller."""

    prompt_tokens: int = 0
    requested_output_tokens: int = 0
    task_category: str = "code"
    complexity_score: float = 0.0
    uncertainty_score: float = 0.0
    wants_verification: bool = False
    latency_target_ms: float | None = None
    queue_pressure: float = 0.0
    memory_pressure: float = 0.0
    backend_tokens_per_second: float = 64.0
    plan_steps: int = 0
    computation_budget: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        """Return the ordered feature vector."""

        category = self.task_category if self.task_category in TASK_CATEGORIES else "code"
        category_flags = {name: (1.0 if name == category else 0.0) for name in TASK_CATEGORIES}
        latency = self.latency_target_ms
        return [
            _clamp(min(self.prompt_tokens, 1024) / 1024.0),
            _clamp(min(self.requested_output_tokens, 1024) / 1024.0),
            category_flags["code"],
            category_flags["explain"],
            category_flags["transform"],
            category_flags["debug"],
            category_flags["structured"],
            _clamp(self.complexity_score),
            _clamp(self.uncertainty_score),
            1.0 if self.wants_verification else 0.0,
            1.0 if latency is not None else 0.0,
            _clamp(min(latency, 120000.0) / 120000.0) if latency is not None else 0.0,
            _clamp(self.queue_pressure),
            _clamp(self.memory_pressure),
            _clamp(min(self.backend_tokens_per_second, 512.0) / 512.0),
            _clamp(min(self.plan_steps, 8) / 8.0),
            _clamp(min(self.computation_budget, 4096) / 4096.0),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "task_category": self.task_category,
            "complexity_score": self.complexity_score,
            "uncertainty_score": self.uncertainty_score,
            "wants_verification": self.wants_verification,
            "latency_target_ms": self.latency_target_ms,
            "queue_pressure": self.queue_pressure,
            "memory_pressure": self.memory_pressure,
            "backend_tokens_per_second": self.backend_tokens_per_second,
            "plan_steps": self.plan_steps,
            "computation_budget": self.computation_budget,
        }


def feature_vector(features: BudgetFeatures | Sequence[float]) -> list[float]:
    """Return the feature vector for either raw features or a vector."""

    if isinstance(features, BudgetFeatures):
        return features.to_vector()
    return [float(value) for value in features]
