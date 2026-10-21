"""Layer 9: budget aware finalisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .controller import BudgetStopReason, RequestBudget

__all__ = ["FinalizationResult", "Finalizer"]


@dataclass
class FinalizationResult:
    """The final response and its truthful metadata."""

    text: str = ""
    finish_reason: str = "stop"
    stop_reason: str = BudgetStopReason.ACCEPTED
    prompt_tokens: int = 0
    completion_tokens: int = 0
    internal_steps: int = 1
    verification_runs: int = 0
    verification_passed: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "internal_steps": self.internal_steps,
            "verification_runs": self.verification_runs,
            "verification_passed": self.verification_passed,
            "notes": list(self.notes),
        }


class Finalizer:
    """Produces the final response of a request."""

    def finalize(
        self,
        text: str,
        budget: RequestBudget,
        finish_reason: str = "stop",
        verification_passed: bool | None = None,
        prompt_tokens: int = 0,
    ) -> FinalizationResult:
        """Finalise a response without exposing internal information."""

        notes: list[str] = []
        if verification_passed is False:
            notes.append(
                "A structural verification check failed; the response was not proven "
                "correct."
            )
        elif verification_passed is True:
            notes.append("The response passed the configured structural checks only.")

        stop_reason = budget.stop_reason or BudgetStopReason.ACCEPTED
        if stop_reason == BudgetStopReason.BUDGET_EXHAUSTED:
            notes.append("The computation budget was exhausted before the response was complete.")

        return FinalizationResult(
            text=text,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=budget.consumed.generation_tokens,
            internal_steps=budget.consumed.internal_steps + 1,
            verification_runs=budget.consumed.verification_passes,
            verification_passed=verification_passed,
            notes=notes,
        )
