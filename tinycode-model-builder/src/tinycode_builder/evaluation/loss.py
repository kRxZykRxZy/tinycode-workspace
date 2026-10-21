"""Loss helpers for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

__all__ = ["LossResult", "sequence_loss", "batched_loss"]


@dataclass
class LossResult:
    """Aggregated loss statistics."""

    loss: float = 0.0
    tokens: int = 0
    sequences: int = 0
    perplexity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "tokens": self.tokens,
            "sequences": self.sequences,
            "perplexity": self.perplexity,
        }


def sequence_loss(model: Any, input_ids: Any, labels: Any = None) -> tuple[float, int]:
    """Compute the loss and token count for a single sequence batch."""

    import torch

    if labels is None:
        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]
    output = model(input_ids, labels=labels)
    if output.loss is None:  # pragma: no cover
        return 0.0, int(input_ids.numel())
    return float(output.loss.item()), int(input_ids.numel())


def batched_loss(
    model: Any, batches: Iterable[Any], device: Any = None, max_batches: int = 0
) -> LossResult:
    """Compute the mean loss over a set of batches."""

    import torch

    total_loss = 0.0
    total_tokens = 0
    sequences = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            input_ids = torch.as_tensor(batch, dtype=torch.long, device=device)
            loss, tokens = sequence_loss(model, input_ids)
            total_loss += loss
            total_tokens += tokens
            sequences += int(input_ids.shape[0])
            count += 1
            if max_batches and count >= max_batches:
                break
    mean_loss = total_loss / count if count else 0.0
    perplexity = float(torch.exp(torch.tensor(mean_loss)).item()) if mean_loss > 0 else 0.0
    return LossResult(
        loss=mean_loss, tokens=total_tokens, sequences=sequences, perplexity=perplexity
    )
