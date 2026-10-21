"""Stop conditions for generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["StopReason", "StopCondition", "StopController"]


class StopReason:
    """The canonical finish reasons."""

    STOP_SEQUENCE = "stop"
    LENGTH = "length"
    EOS_TOKEN = "eos"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE = "deadline"
    ERROR = "error"


@dataclass
class StopCondition:
    """The conditions that end a generation."""

    max_tokens: int = 128
    stop_sequences: tuple[str, ...] = ()
    eos_token_id: int | None = None
    deadline: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "stop_sequences": list(self.stop_sequences),
            "eos_token_id": self.eos_token_id,
            "deadline": self.deadline,
        }


class StopController:
    """Evaluates the stop conditions while generating."""

    def __init__(self, condition: StopCondition) -> None:
        self.condition = condition
        self.generated = 0
        self.stop_reason: str | None = None

    def check_length(self) -> str | None:
        if self.generated >= self.condition.max_tokens:
            self.stop_reason = StopReason.LENGTH
            return self.stop_reason
        return None

    def check_eos(self, token_id: int) -> str | None:
        if self.condition.eos_token_id is not None and token_id == self.condition.eos_token_id:
            self.stop_reason = StopReason.EOS_TOKEN
            return self.stop_reason
        return None

    def check_stop_sequences(self, text: str) -> str | None:
        for sequence in self.condition.stop_sequences:
            if sequence and sequence in text:
                self.stop_reason = StopReason.STOP_SEQUENCE
                return self.stop_reason
        return None

    def check_deadline(self, now: float) -> str | None:
        if self.condition.deadline is not None and now >= self.condition.deadline:
            self.stop_reason = StopReason.DEADLINE
            return self.stop_reason
        return None

    def update(self, token_id: int, text: str, now: float) -> str | None:
        """Update the controller and return a stop reason when generation must end."""

        self.generated += 1
        for reason in (
            self.check_length(),
            self.check_eos(token_id),
            self.check_stop_sequences(text),
            self.check_deadline(now),
        ):
            if reason:
                return reason
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "stop_reason": self.stop_reason,
            "condition": self.condition.to_dict(),
        }
