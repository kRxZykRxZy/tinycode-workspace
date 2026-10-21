"""Adaptive micro batch sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["AdaptiveBatchPolicy", "choose_micro_batch"]


@dataclass
class AdaptiveBatchPolicy:
    """Bounds and targets for the micro batch size."""

    min_micro_batch: int = 1
    max_micro_batch: int = 64
    target_tokens_per_micro_batch: int = 4096
    sequence_length: int = 1024

    def max_for_tokens(self) -> int:
        return max(1, self.target_tokens_per_micro_batch // max(1, self.sequence_length))


def choose_micro_batch(
    policy: AdaptiveBatchPolicy,
    available_memory_bytes: int | None,
    estimated_bytes_per_sequence: int,
    current: int = 1,
    memory_fraction: float = 0.5,
) -> int:
    """Choose a micro batch size that fits into the available memory.

    The function is deterministic and never returns a value outside of the
    configured bounds.  It never changes the sequence length.
    """

    upper = min(policy.max_micro_batch, policy.max_for_tokens())
    lower = max(1, policy.min_micro_batch)

    if available_memory_bytes is None or estimated_bytes_per_sequence <= 0:
        return max(lower, min(upper, current or lower))

    budget = int(available_memory_bytes * max(0.05, min(1.0, memory_fraction)))
    affordable = max(1, budget // estimated_bytes_per_sequence)
    chosen = int(min(upper, affordable))
    if chosen < lower:
        chosen = lower
    # Never grow the batch by more than 2x in one step (stability).
    if current and chosen > current * 2:
        chosen = current * 2
    return int(max(lower, min(upper, chosen)))


def estimate_bytes_per_sequence(
    parameter_count: int,
    d_model: int,
    n_layers: int,
    sequence_length: int,
    precision: str = "float32",
) -> int:
    """Rough estimate of the activation memory used by one sequence."""

    from .memory_estimator import dtype_bytes

    element = dtype_bytes(precision)
    return max(1, sequence_length * d_model * n_layers * element * 2)
