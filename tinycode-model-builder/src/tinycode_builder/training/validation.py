"""Validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

__all__ = ["ValidationResult", "evaluate_loss", "iter_validation_batches"]


@dataclass
class ValidationResult:
    """Aggregated validation metrics."""

    loss: float = 0.0
    tokens: int = 0
    batches: int = 0
    perplexity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "tokens": self.tokens,
            "batches": self.batches,
            "perplexity": self.perplexity,
        }


def iter_validation_batches(shards: Iterable[Any], batch_size: int, limit: int = 0) -> Iterable[Any]:
    """Yield validation batches from packed shards."""

    produced = 0
    for shard in shards:
        array = shard
        sequences = array.shape[0]
        for start in range(0, sequences - batch_size + 1, batch_size):
            window = array[start : start + batch_size]
            if window.shape[0] == 0:
                continue
            yield window
            produced += 1
            if limit and produced >= limit:
                return


def evaluate_loss(
    model: Any,
    batches: Iterable[Any],
    device: Any,
    precision_plan: Any = None,
    autocast_factory: Any = None,
    max_batches: int = 0,
) -> ValidationResult:
    """Compute the mean validation loss."""

    import torch

    from .precision import autocast_context

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    count = 0
    with torch.no_grad():
        for batch in batches:
            input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
            labels = torch.full_like(input_ids, -100)
            labels[:, :-1] = input_ids[:, 1:]
            context = (
                autocast_factory(precision_plan)
                if autocast_factory is not None
                else autocast_context(precision_plan)
            )
            with context:
                output = model(input_ids, labels=labels)
            loss = output.loss
            if loss is None:  # pragma: no cover - labels always provided
                continue
            total_loss += float(loss.item())
            total_tokens += int(input_ids.numel())
            count += 1
            if max_batches and count >= max_batches:
                break
    model.train()
    mean_loss = total_loss / count if count else 0.0
    perplexity = float(torch.exp(torch.tensor(mean_loss)).item()) if mean_loss > 0 else 0.0
    return ValidationResult(loss=mean_loss, tokens=total_tokens, batches=count, perplexity=perplexity)
