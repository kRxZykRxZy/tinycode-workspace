"""Repetition and frequency penalties."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["apply_repetition_penalty", "apply_frequency_penalty", "apply_presence_penalty"]


def apply_repetition_penalty(logits: Any, generated: Sequence[int], penalty: float) -> Any:
    """Penalise tokens that already appeared in the output."""

    if not penalty or not generated:
        return logits
    values = np.array(logits, copy=True)
    unique = set(int(token) for token in generated)
    for token in unique:
        if 0 <= token < values.shape[-1]:
            if values[token] > 0:
                values[token] -= penalty
            else:
                values[token] += penalty
    return values


def apply_frequency_penalty(logits: Any, generated: Sequence[int], penalty: float) -> Any:
    """Penalise tokens proportionally to how often they occurred."""

    if not penalty or not generated:
        return logits
    values = np.array(logits, copy=True)
    counts: dict[int, int] = {}
    for token in generated:
        counts[int(token)] = counts.get(int(token), 0) + 1
    for token, count in counts.items():
        if 0 <= token < values.shape[-1]:
            values[token] -= penalty * count
    return values


def apply_presence_penalty(logits: Any, generated: Sequence[int], penalty: float) -> Any:
    """Penalise tokens that occurred at least once."""

    if not penalty or not generated:
        return logits
    values = np.array(logits, copy=True)
    for token in set(int(item) for item in generated):
        if 0 <= token < values.shape[-1]:
            values[token] -= penalty
    return values
