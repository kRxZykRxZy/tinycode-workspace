"""Logits processing."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .repetition import (
    apply_frequency_penalty,
    apply_presence_penalty,
    apply_repetition_penalty,
)

__all__ = ["apply_logits_processors", "LogitsProcessors"]


class LogitsProcessors:
    """The penalties that are applied before a token is selected."""

    def __init__(
        self,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> None:
        self.repetition_penalty = float(repetition_penalty)
        self.frequency_penalty = float(frequency_penalty)
        self.presence_penalty = float(presence_penalty)

    def apply(self, logits: Any, generated: Sequence[int]) -> Any:
        """Apply every configured penalty."""

        values = np.asarray(logits, dtype=np.float32)
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            values = apply_repetition_penalty(values, generated, self.repetition_penalty)
        if self.frequency_penalty:
            values = apply_frequency_penalty(values, generated, self.frequency_penalty)
        if self.presence_penalty:
            values = apply_presence_penalty(values, generated, self.presence_penalty)
        return values

    def to_dict(self) -> dict[str, float]:
        return {
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


def apply_logits_processors(
    logits: Any,
    generated: Sequence[int],
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Any:
    """Functional interface of :class:`LogitsProcessors`."""

    processors = LogitsProcessors(repetition_penalty, frequency_penalty, presence_penalty)
    return processors.apply(logits, generated)
