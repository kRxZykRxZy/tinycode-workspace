"""A single optimizer step (including gradient accumulation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["TrainStepResult", "run_train_step", "build_batch_tensors"]


@dataclass
class TrainStepResult:
    """Result of one optimizer step."""

    loss: float = 0.0
    tokens: int = 0
    micro_batches: int = 0
    gradient_norm: float | None = None
    learning_rate: float = 0.0
    skipped: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "tokens": self.tokens,
            "micro_batches": self.micro_batches,
            "gradient_norm": self.gradient_norm,
            "learning_rate": self.learning_rate,
            "skipped": self.skipped,
            "reason": self.reason,
        }


def build_batch_tensors(batch: Any, device: Any) -> tuple[Any, Any]:
    """Build ``(input_ids, labels)`` tensors from a packed sequence batch.

    The labels are the inputs shifted by one position; the last position
    is ignored.  The input length stays at the full 1024 token context.
    """

    import numpy as np
    import torch

    # The shards are memory mapped and therefore read only; copy the batch
    # so that torch never works on a non writable buffer.
    input_ids = torch.as_tensor(np.array(batch, copy=True), dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:]
    return input_ids, labels


def run_train_step(
    model: Any,
    batch: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    plan: Any,
    device: Any,
    grad_clip: float | None = 1.0,
    micro_batch_size: int | None = None,
) -> TrainStepResult:
    """Run one optimizer step over ``batch``.

    The batch is split into micro batches; gradients are accumulated and a
    single optimizer step is taken once all micro batches are processed.
    """

    import torch

    from .gradient_accumulation import scale_loss, unscale_and_clip
    from .precision import autocast_context

    micro_batch_size = micro_batch_size or getattr(plan, "micro_batch_size", 1)

    input_ids, labels = build_batch_tensors(batch, device)
    total_tokens = int(input_ids.numel())

    optimizer.zero_grad(set_to_none=True)

    micro_batches = 0
    loss_sum = 0.0
    total_micro = max(1, (input_ids.shape[0] + micro_batch_size - 1) // micro_batch_size)

    for start in range(0, input_ids.shape[0], micro_batch_size):
        micro_inputs = input_ids[start : start + micro_batch_size]
        micro_labels = labels[start : start + micro_batch_size]
        context = autocast_context(plan)
        with context:
            output = model(micro_inputs, labels=micro_labels)
            loss = output.loss
            if loss is None:  # pragma: no cover - labels always provided
                continue
            scaled = scale_loss(loss, total_micro)
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
        loss_sum += float(loss.detach().item())
        micro_batches += 1

    gradient_norm = unscale_and_clip(optimizer, scaler, grad_clip)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    learning_rate = scheduler.step() if scheduler is not None else 0.0

    return TrainStepResult(
        loss=loss_sum / micro_batches if micro_batches else 0.0,
        tokens=total_tokens,
        micro_batches=micro_batches,
        gradient_norm=gradient_norm,
        learning_rate=learning_rate,
    )
