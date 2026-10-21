"""Perplexity measurement."""

from __future__ import annotations

from typing import Any, Iterable

from .loss import LossResult, batched_loss

__all__ = ["compute_perplexity", "perplexity_from_loss"]


def perplexity_from_loss(loss: float) -> float:
    """Convert a mean cross entropy into perplexity."""

    import math

    if loss <= 0:
        return 0.0
    return float(math.exp(loss))


def compute_perplexity(
    model: Any,
    batches: Iterable[Any],
    device: Any = None,
    max_batches: int = 0,
) -> LossResult:
    """Measure the perplexity of a model over packed batches."""

    result = batched_loss(model, batches, device=device, max_batches=max_batches)
    result.perplexity = perplexity_from_loss(result.loss)
    return result
