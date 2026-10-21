"""The adaptive budget controller and the request budget objects.

The engine always has a deterministic baseline controller.  A trained
controller may be attached from the artifact; when it is missing or
incompatible the baseline is used and the engine keeps working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "BudgetFeatures",
    "BudgetLimits",
    "BudgetAllocation",
    "RequestBudget",
    "BudgetStopReason",
    "ControllerDecision",
    "BudgetController",
    "BaselineBudgetController",
    "TrainedBudgetController",
    "feature_vector",
]

#: The feature schema version shared with the model builder.
FEATURE_SCHEMA_VERSION = "budget_features.v1"

#: The feature order is part of the contract; never reorder it.
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

TASK_CATEGORIES: tuple[str, ...] = ("code", "explain", "transform", "debug", "structured")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


@dataclass
class BudgetFeatures:
    """The inputs of the budget controller."""

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

    def to_vector(self) -> list[float]:
        """Return the ordered feature vector."""

        category = self.task_category if self.task_category in TASK_CATEGORIES else "code"
        flags = {name: (1.0 if name == category else 0.0) for name in TASK_CATEGORIES}
        latency = self.latency_target_ms
        return [
            _clamp(min(self.prompt_tokens, 1024) / 1024.0),
            _clamp(min(self.requested_output_tokens, 1024) / 1024.0),
            flags["code"],
            flags["explain"],
            flags["transform"],
            flags["debug"],
            flags["structured"],
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
    """Return the feature vector of a feature object or a raw vector."""

    if isinstance(features, BudgetFeatures):
        return features.to_vector()
    return [float(value) for value in features]


class BudgetStopReason:
    """Why the budget layer stopped a request."""

    ACCEPTED = "accepted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE_REACHED = "deadline_reached"
    CONFIDENCE_SUFFICIENT = "confidence_sufficient"
    NO_MARGINAL_GAIN = "no_marginal_gain"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class BudgetLimits:
    """Server side limits; a user budget can never exceed them."""

    max_generation_tokens: int = 1024
    max_internal_steps: int = 8
    max_verification_passes: int = 2
    max_computation_tokens: int = 4096
    max_latency_ms: int = 120_000
    max_prompt_tokens: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_generation_tokens": self.max_generation_tokens,
            "max_internal_steps": self.max_internal_steps,
            "max_verification_passes": self.max_verification_passes,
            "max_computation_tokens": self.max_computation_tokens,
            "max_latency_ms": self.max_latency_ms,
            "max_prompt_tokens": self.max_prompt_tokens,
        }


@dataclass
class BudgetAllocation:
    """The computation budget that was granted to a request."""

    policy: str = "standard"
    generation_tokens: int = 256
    internal_steps: int = 1
    verification_passes: int = 0
    allow_planning: bool = False
    deadline_ms: float | None = None
    estimated_memory_bytes: int = 0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "generation_tokens": self.generation_tokens,
            "internal_steps": self.internal_steps,
            "verification_passes": self.verification_passes,
            "allow_planning": self.allow_planning,
            "deadline_ms": self.deadline_ms,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "rationale": self.rationale,
        }


@dataclass
class RequestBudget:
    """The complete lifecycle of the computation budget of one request."""

    request_id: str = ""
    task_category: str = "code"
    complexity: float = 0.0
    uncertainty: float = 0.0
    requested_latency_ms: float | None = None
    requested_compute_budget: int | None = None
    max_generation_tokens: int = 256
    max_internal_steps: int = 1
    verification_allowance: int = 0
    deadline: float | None = None
    memory_estimate_bytes: int = 0
    selected_policy: str = "standard"
    allocation: BudgetAllocation = field(default_factory=BudgetAllocation)
    consumed: BudgetAllocation = field(default_factory=BudgetAllocation)
    stop_reason: str = ""
    limits: BudgetLimits = field(default_factory=BudgetLimits)

    @property
    def remaining_generation_tokens(self) -> int:
        return max(0, self.allocation.generation_tokens - self.consumed.generation_tokens)

    @property
    def remaining_steps(self) -> int:
        return max(0, self.allocation.internal_steps - self.consumed.internal_steps)

    @property
    def remaining_verification_passes(self) -> int:
        return max(0, self.allocation.verification_passes - self.consumed.verification_passes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_category": self.task_category,
            "complexity": self.complexity,
            "uncertainty": self.uncertainty,
            "requested_latency_ms": self.requested_latency_ms,
            "requested_compute_budget": self.requested_compute_budget,
            "max_generation_tokens": self.max_generation_tokens,
            "max_internal_steps": self.max_internal_steps,
            "verification_allowance": self.verification_allowance,
            "deadline": self.deadline,
            "memory_estimate_bytes": self.memory_estimate_bytes,
            "selected_policy": self.selected_policy,
            "allocation": self.allocation.to_dict(),
            "consumed": self.consumed.to_dict(),
            "stop_reason": self.stop_reason,
            "limits": self.limits.to_dict(),
        }


@dataclass
class ControllerDecision:
    """The recommendation of the budget controller."""

    policy: str = "standard"
    generation_tokens: int = 256
    internal_steps: int = 1
    verification_passes: int = 0
    allow_planning: bool = False
    confidence: float = 0.0
    rationale: str = ""
    source: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "generation_tokens": self.generation_tokens,
            "internal_steps": self.internal_steps,
            "verification_passes": self.verification_passes,
            "allow_planning": self.allow_planning,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source": self.source,
        }


class BudgetController(Protocol):
    """The interface of a budget controller."""

    def decide(self, features: BudgetFeatures, requested_tokens: int) -> ControllerDecision:
        ...

    def describe(self) -> dict[str, Any]:
        ...


class BaselineBudgetController:
    """A deterministic, inspectable budget policy."""

    version = "1.0.0"
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def decide(self, features: BudgetFeatures, requested_tokens: int) -> ControllerDecision:
        complexity = max(0.0, min(1.0, features.complexity_score))
        uncertainty = max(0.0, min(1.0, features.uncertainty_score))

        if complexity < 0.25:
            policy = "minimal"
        elif complexity < 0.55:
            policy = "standard"
        else:
            policy = "extended"

        wants_verification = bool(features.wants_verification) or uncertainty >= 0.6
        if wants_verification and complexity >= 0.55:
            policy = "verification"

        bounds = {
            "minimal": (32, 128, 1, False, 0),
            "standard": (64, 384, 2, False, 0),
            "extended": (128, 768, 4, True, 0),
            "verification": (96, 640, 4, True, 1),
        }
        minimum, maximum, steps, planning, verification = bounds[policy]

        tokens = int(max(minimum, min(maximum, requested_tokens)))
        if features.latency_target_ms is not None:
            scale = min(1.0, features.latency_target_ms / 30_000.0)
            tokens = int(max(minimum, tokens * scale))

        internal_steps = max(1, min(steps, features.plan_steps + 1))

        return ControllerDecision(
            policy=policy,
            generation_tokens=tokens,
            internal_steps=internal_steps,
            verification_passes=verification,
            allow_planning=planning and complexity >= 0.55,
            confidence=round(1.0 - uncertainty, 4),
            rationale=(
                f"baseline policy: complexity={complexity:.2f} uncertainty={uncertainty:.2f} "
                f"-> {policy}"
            ),
            source="baseline",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "baseline",
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
        }


class TrainedBudgetController:
    """A controller backed by the trained weights of the artifact.

    The weights are stored as a torch state dictionary.  The controller
    refuses to load weights that do not match the feature schema, and the
    caller falls back to the baseline policy when that happens.
    """

    version = "1.0.0"

    def __init__(
        self,
        state_dict: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = dict(metadata or {})
        self.metadata = metadata
        self.feature_schema_version = str(
            metadata.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        )
        self.feature_names = list(metadata.get("feature_names", FEATURE_NAMES))
        self.hidden_sizes = list(metadata.get("hidden_sizes", [64, 32]))
        self.state_dict = state_dict
        self._model: Any = None

    # ------------------------------------------------------------------
    @property
    def compatible(self) -> bool:
        return self.feature_schema_version == FEATURE_SCHEMA_VERSION and list(
            self.feature_names
        ) == list(FEATURE_NAMES)

    def _build(self) -> Any:
        import torch
        from torch import nn

        layers: list[nn.Module] = []
        previous = len(self.feature_names)
        for size in self.hidden_sizes:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.Tanh())
            previous = size
        layers.append(nn.Linear(previous, 4))
        model = nn.Sequential(*layers)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def predict(self, features: BudgetFeatures) -> dict[str, float]:
        """Return the raw controller outputs."""

        import torch

        if self._model is None:
            self._model = self._build()
        vector = torch.tensor([features.to_vector()], dtype=torch.float32)
        with torch.no_grad():
            outputs = self._model(vector)[0].tolist()
        return {
            "budget_multiplier": float(outputs[0]),
            "planning_logit": float(outputs[1]),
            "verification_logit": float(outputs[2]),
            "stop_logit": float(outputs[3]),
        }

    def decide(self, features: BudgetFeatures, requested_tokens: int) -> ControllerDecision:
        if not self.compatible:
            baseline = BaselineBudgetController().decide(features, requested_tokens)
            baseline.rationale = "trained controller is incompatible; used the baseline policy"
            return baseline
        try:
            raw = self.predict(features)
        except Exception:
            baseline = BaselineBudgetController().decide(features, requested_tokens)
            baseline.rationale = "trained controller failed; used the baseline policy"
            return baseline

        multiplier = max(0.25, min(4.0, raw["budget_multiplier"]))
        planning = raw["planning_logit"] > 0.0
        verification = raw["verification_logit"] > 0.0
        tokens = int(max(32, min(1024, requested_tokens * multiplier)))
        steps = 1 + (1 if planning else 0) + (1 if verification else 0)

        return ControllerDecision(
            policy="verification" if verification else ("extended" if planning else "standard"),
            generation_tokens=tokens,
            internal_steps=min(steps, 8),
            verification_passes=1 if verification else 0,
            allow_planning=planning,
            confidence=max(0.0, min(1.0, 1.0 - features.uncertainty_score)),
            rationale=(
                f"trained controller: multiplier={multiplier:.2f} plan={planning} "
                f"verify={verification}"
            ),
            source="trained",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "trained",
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "compatible": self.compatible,
            "hidden_sizes": self.hidden_sizes,
        }
