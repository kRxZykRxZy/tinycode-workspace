"""Gradient accumulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

__all__ = ["AccumulationPlan", "scale_loss", "unscale_and_clip"]


@dataclass
class AccumulationPlan:
    """How a batch is split into micro batches."""

    accumulation_steps: int = 1
    micro_batch_size: int = 1

    @property
    def effective_batch_size(self) -> int:
        return self.accumulation_steps * self.micro_batch_size


def scale_loss(loss: Any, accumulation_steps: int) -> Any:
    """Scale a loss so that gradients accumulate correctly."""

    if accumulation_steps <= 1:
        return loss
    return loss / float(accumulation_steps)


def unscale_and_clip(
    optimizer: Any, scaler: Any, max_norm: float | None
) -> float | None:
    """Unscale gradients (when a scaler is used) and clip them.

    Returns:
        The total gradient norm before clipping, or ``None`` when no
        gradients were present.
    """

    if max_norm is not None and max_norm > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
    total_norm = None
    if max_norm is not None and max_norm > 0:
        total_norm = float(
            __import__("torch").nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                max_norm,
            )
        )
    return total_norm


def iter_microbatches(batch: Any, micro_batch_size: int) -> Iterator[Any]:
    """Split a batch tensor along the batch dimension."""

    if micro_batch_size <= 0:
        yield batch
        return
    total = batch.shape[0]
    for start in range(0, total, micro_batch_size):
        yield batch[start : start + micro_batch_size]
