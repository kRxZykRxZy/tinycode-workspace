"""Layer 3: initial budget allocation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .complexity_estimator import ComplexityEstimate
from .controller import (
    BaselineBudgetController,
    BudgetAllocation,
    BudgetController,
    BudgetFeatures,
    BudgetLimits,
    BudgetStopReason,
    ControllerDecision,
    RequestBudget,
)
from .request_analyzer import AnalyzedRequest

__all__ = ["ResourceState", "BudgetAllocator"]


@dataclass
class ResourceState:
    """The state of the runtime that influences the allocation."""

    queue_depth: int = 0
    queue_capacity: int = 64
    memory_pressure: float = 0.0
    backend_tokens_per_second: float = 64.0
    available_memory_bytes: int = 0
    active_requests: int = 0

    @property
    def queue_pressure(self) -> float:
        if self.queue_capacity <= 0:
            return 0.0
        return min(1.0, self.queue_depth / float(self.queue_capacity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_depth": self.queue_depth,
            "queue_capacity": self.queue_capacity,
            "queue_pressure": round(self.queue_pressure, 4),
            "memory_pressure": round(self.memory_pressure, 4),
            "backend_tokens_per_second": self.backend_tokens_per_second,
            "available_memory_bytes": self.available_memory_bytes,
            "active_requests": self.active_requests,
        }


class BudgetAllocator:
    """Converts the complexity estimate and the request constraints into a budget."""

    def __init__(
        self,
        controller: BudgetController | None = None,
        limits: BudgetLimits | None = None,
    ) -> None:
        self.controller = controller or BaselineBudgetController()
        self.limits = limits or BudgetLimits()

    # ------------------------------------------------------------------
    def build_features(
        self,
        request: AnalyzedRequest,
        complexity: ComplexityEstimate,
        uncertainty: float,
        resources: ResourceState,
        plan_steps: int = 0,
    ) -> BudgetFeatures:
        """Build the controller input features."""

        return BudgetFeatures(
            prompt_tokens=request.prompt_tokens,
            requested_output_tokens=request.requested_output_tokens,
            task_category=request.task_category,
            complexity_score=complexity.score,
            uncertainty_score=uncertainty,
            wants_verification=request.wants_verification,
            latency_target_ms=request.latency_target_ms,
            queue_pressure=resources.queue_pressure,
            memory_pressure=resources.memory_pressure,
            backend_tokens_per_second=resources.backend_tokens_per_second,
            plan_steps=plan_steps,
            computation_budget=self.limits.max_computation_tokens,
        )

    # ------------------------------------------------------------------
    def allocate(
        self,
        request: AnalyzedRequest,
        complexity: ComplexityEstimate,
        uncertainty: float,
        resources: ResourceState,
        now: float | None = None,
    ) -> RequestBudget:
        """Allocate the initial computation budget of a request."""

        features = self.build_features(request, complexity, uncertainty, resources)
        decision: ControllerDecision = self.controller.decide(
            features, request.requested_output_tokens
        )

        generation_tokens = self._clamp_generation(decision.generation_tokens)
        internal_steps = max(1, min(decision.internal_steps, self.limits.max_internal_steps))
        verification_passes = max(
            0, min(decision.verification_passes, self.limits.max_verification_passes)
        )

        requested_budget = request.requested_compute_budget
        if requested_budget is not None:
            generation_tokens = min(generation_tokens, max(1, requested_budget))
            internal_steps = min(internal_steps, max(1, requested_budget // 64))

        # Under queue or memory pressure the allocation shrinks; the
        # request still runs, but with less compute.
        pressure = max(resources.queue_pressure, resources.memory_pressure)
        if pressure > 0.5:
            generation_tokens = max(1, int(generation_tokens * (1.0 - 0.5 * pressure)))
            internal_steps = max(1, internal_steps - 1 if pressure > 0.8 else internal_steps)

        deadline_ms: float | None = None
        if request.latency_target_ms is not None:
            deadline_ms = float(request.latency_target_ms)
        elif resources.backend_tokens_per_second > 0:
            estimate_ms = generation_tokens / resources.backend_tokens_per_second * 1000.0
            deadline_ms = min(float(self.limits.max_latency_ms), max(1000.0, estimate_ms * 4.0))

        now = now if now is not None else time.time()

        allocation = BudgetAllocation(
            policy=decision.policy,
            generation_tokens=generation_tokens,
            internal_steps=internal_steps,
            verification_passes=verification_passes,
            allow_planning=decision.allow_planning,
            deadline_ms=deadline_ms,
            estimated_memory_bytes=self._estimate_memory(request, generation_tokens),
            rationale=decision.rationale,
        )

        budget = RequestBudget(
            request_id=request.request_id,
            task_category=request.task_category,
            complexity=complexity.score,
            uncertainty=uncertainty,
            requested_latency_ms=request.latency_target_ms,
            requested_compute_budget=requested_budget,
            max_generation_tokens=generation_tokens,
            max_internal_steps=internal_steps,
            verification_allowance=verification_passes,
            deadline=(now + deadline_ms / 1000.0) if deadline_ms else None,
            memory_estimate_bytes=allocation.estimated_memory_bytes,
            selected_policy=decision.policy,
            allocation=allocation,
            limits=self.limits,
            stop_reason=BudgetStopReason.ACCEPTED,
        )
        return budget

    # ------------------------------------------------------------------
    def _clamp_generation(self, tokens: int) -> int:
        return int(max(1, min(self.limits.max_generation_tokens, tokens)))

    def _estimate_memory(self, request: AnalyzedRequest, generation_tokens: int) -> int:
        """Rough memory estimate of the KV cache of a request."""

        # A conservative estimate: 4 bytes per token per layer for the key
        # and value vectors of a moderate hidden size.
        per_token = 4 * 2 * 512
        return int((request.prompt_tokens + generation_tokens) * per_token)
