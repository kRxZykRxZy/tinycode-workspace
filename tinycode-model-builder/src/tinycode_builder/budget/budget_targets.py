"""Request budget objects.

These structures mirror the engine side request budget so that offline
policy training and evaluation use the same vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "StopReason",
    "RequestBudget",
    "BudgetLimits",
    "BudgetAllocation",
]


class StopReason:
    """Why a request stopped consuming computation."""

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
    """Server side limits that user requests may not exceed."""

    max_generation_tokens: int = 1024
    max_internal_steps: int = 8
    max_verification_passes: int = 2
    max_computation_tokens: int = 4096
    max_latency_ms: int = 120_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_generation_tokens": self.max_generation_tokens,
            "max_internal_steps": self.max_internal_steps,
            "max_verification_passes": self.max_verification_passes,
            "max_computation_tokens": self.max_computation_tokens,
            "max_latency_ms": self.max_latency_ms,
        }


@dataclass
class BudgetAllocation:
    """The budget that was granted to a request."""

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
    """The full lifecycle of the computation budget of a single request."""

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
