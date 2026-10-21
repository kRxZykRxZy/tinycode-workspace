"""The execution controller: runs the model inside the granted budget."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..budget.confidence import ConfidenceEstimator, UncertaintySignals
from ..budget.controller import BudgetStopReason, RequestBudget
from ..budget.scheduler import BudgetScheduler, ContinuationDecision
from ..generation.generator import GenerationRequest, GenerationResult

__all__ = ["ExecutionOutcome", "ExecutionController"]


@dataclass
class ExecutionOutcome:
    """The result of executing a request."""

    result: GenerationResult | None = None
    signals: UncertaintySignals = field(default_factory=UncertaintySignals)
    decision: ContinuationDecision | None = None
    passes: int = 1
    elapsed_ms: float = 0.0
    cancelled: bool = False

    @property
    def text(self) -> str:
        return self.result.text if self.result else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "cancelled": self.cancelled,
            "signals": self.signals.to_dict(),
            "decision": self.decision.to_dict() if self.decision else None,
        }


class ExecutionController:
    """Runs the generation and the adaptive decisions."""

    def __init__(
        self,
        generator: Any,
        scheduler: BudgetScheduler | None = None,
        confidence: ConfidenceEstimator | None = None,
    ) -> None:
        self.generator = generator
        self.scheduler = scheduler or BudgetScheduler()
        self.confidence = confidence or ConfidenceEstimator()

    # ------------------------------------------------------------------
    def run(
        self,
        request: GenerationRequest,
        budget: RequestBudget,
        cancelled: Callable[[], bool] | None = None,
        on_token: Callable[[Any], None] | None = None,
        collect_logits: bool = True,
        max_passes: int = 2,
        prefix: Any = None,
    ) -> ExecutionOutcome:
        """Execute the request within the granted budget."""

        started = time.perf_counter()
        outcome = ExecutionOutcome()

        logits_history: list[Any] = []
        token_ids: list[int] = []

        def _on_logits(logits: Any, generated: list[int]) -> Any:
            if collect_logits:
                logits_history.append(logits)
            token_ids.clear()
            token_ids.extend(generated)
            return logits

        passes = 0
        while True:
            passes += 1
            remaining = budget.remaining_generation_tokens
            if remaining <= 0:
                outcome.decision = ContinuationDecision(
                    action="stop_budget_exhausted",
                    reason="the computation budget is exhausted",
                )
                budget.stop_reason = BudgetStopReason.BUDGET_EXHAUSTED
                break

            request.max_tokens = min(request.max_tokens, remaining)
            result = self.generator.generate(
                request,
                cancelled=cancelled,
                on_token=on_token if passes == 1 else None,
                on_logits=_on_logits,
                prefix=prefix if passes == 1 else None,
            )
            outcome.result = result
            budget.consumed.generation_tokens += result.completion_tokens
            budget.consumed.internal_steps = max(budget.consumed.internal_steps, passes)

            if result.finish_reason == "cancelled":
                outcome.cancelled = True
                budget.stop_reason = BudgetStopReason.CANCELLED
                break

            signals = self.confidence.from_logits(logits_history)
            outcome.signals = signals

            decision = self.scheduler.decide(
                budget,
                signals,
                now=time.time(),
                passes_completed=passes - 1,
            )
            outcome.decision = decision
            self.scheduler.apply(budget, decision)

            if decision.action == "accept":
                budget.stop_reason = BudgetStopReason.CONFIDENCE_SUFFICIENT
                break
            if decision.action == "additional_pass":
                if passes >= max_passes:
                    budget.stop_reason = BudgetStopReason.NO_MARGINAL_GAIN
                    break
                continue
            if decision.action == "verify":
                # The verification stage is executed by the request
                # pipeline; the budget records the pass.
                continue
            budget.stop_reason = self.scheduler.stop_reason_for(decision) or BudgetStopReason.ACCEPTED
            break

        outcome.passes = passes
        outcome.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return outcome
