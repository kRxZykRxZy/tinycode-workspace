"""Layer 7: the adaptive computation decision.

The decision is made with explicit rules and bounded costs.  A learned
policy can replace the deterministic rules, but the loop is always finite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .confidence import UncertaintySignals
from .controller import BudgetStopReason, RequestBudget

__all__ = ["ContinuationDecision", "BudgetScheduler", "DECISIONS"]

DECISIONS = (
    "accept",
    "continue_generation",
    "additional_pass",
    "verify",
    "stop_budget_exhausted",
    "stop_deadline",
    "stop_no_marginal_gain",
)


@dataclass
class ContinuationDecision:
    """What the scheduler decided to do next."""

    action: str = "accept"
    reason: str = ""
    remaining_tokens: int = 0
    remaining_steps: int = 0
    remaining_verification_passes: int = 0
    confidence: float = 0.0

    @property
    def should_continue(self) -> bool:
        return self.action in {"continue_generation", "additional_pass", "verify"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "remaining_tokens": self.remaining_tokens,
            "remaining_steps": self.remaining_steps,
            "remaining_verification_passes": self.remaining_verification_passes,
            "confidence": round(self.confidence, 6),
        }


class BudgetScheduler:
    """Decides whether a request should receive more computation."""

    def __init__(
        self,
        uncertainty_threshold: float = 0.55,
        target_confidence: float = 0.7,
        max_passes: int = 2,
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.target_confidence = target_confidence
        self.max_passes = max_passes

    # ------------------------------------------------------------------
    def decide(
        self,
        budget: RequestBudget,
        signals: UncertaintySignals,
        now: float | None = None,
        passes_completed: int = 0,
    ) -> ContinuationDecision:
        """Decide whether to continue, verify or stop."""

        remaining_tokens = budget.remaining_generation_tokens
        remaining_steps = budget.remaining_steps
        remaining_verifications = budget.remaining_verification_passes
        confidence = 1.0 - signals.score

        decision = ContinuationDecision(
            action="accept",
            reason="",
            remaining_tokens=remaining_tokens,
            remaining_steps=remaining_steps,
            remaining_verification_passes=remaining_verifications,
            confidence=confidence,
        )

        if budget.deadline is not None and now is not None and now >= budget.deadline:
            decision.action = "stop_deadline"
            decision.reason = "the request deadline was reached"
            return decision

        if remaining_tokens <= 0 and remaining_steps <= 0 and remaining_verifications <= 0:
            decision.action = "stop_budget_exhausted"
            decision.reason = "the computation budget is exhausted"
            return decision

        if confidence >= self.target_confidence:
            decision.action = "accept"
            decision.reason = (
                f"the confidence estimate ({confidence:.2f}) reached the target "
                f"({self.target_confidence:.2f})"
            )
            return decision

        if signals.verification_failed and remaining_verifications > 0:
            decision.action = "verify"
            decision.reason = "the previous verification run failed and budget remains"
            return decision

        if signals.score >= self.uncertainty_threshold and remaining_steps > 0:
            if passes_completed >= self.max_passes:
                decision.action = "stop_no_marginal_gain"
                decision.reason = (
                    "additional passes are unlikely to help; the maximum number of "
                    "supported passes was reached"
                )
                return decision
            decision.action = "additional_pass"
            decision.reason = (
                f"the uncertainty estimate ({signals.score:.2f}) is above the "
                f"threshold ({self.uncertainty_threshold:.2f})"
            )
            return decision

        if remaining_tokens > 0 and remaining_steps > 0:
            decision.action = "continue_generation"
            decision.reason = "the budget still allows more generation"
            return decision

        decision.action = "accept"
        decision.reason = "no further computation is justified"
        return decision

    # ------------------------------------------------------------------
    def apply(self, budget: RequestBudget, decision: ContinuationDecision) -> None:
        """Record the decision in the request budget."""

        if decision.action == "verify":
            budget.consumed.verification_passes += 1
        elif decision.action in {"continue_generation", "additional_pass"}:
            budget.consumed.internal_steps += 1

    def stop_reason_for(self, decision: ContinuationDecision) -> str:
        """Map a decision onto a budget stop reason."""

        if decision.action == "stop_deadline":
            return BudgetStopReason.DEADLINE_REACHED
        if decision.action == "stop_budget_exhausted":
            return BudgetStopReason.BUDGET_EXHAUSTED
        if decision.action == "stop_no_marginal_gain":
            return BudgetStopReason.NO_MARGINAL_GAIN
        if decision.action == "accept":
            return BudgetStopReason.CONFIDENCE_SUFFICIENT
        return ""
