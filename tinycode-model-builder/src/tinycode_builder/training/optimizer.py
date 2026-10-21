"""Optimizer construction and parameter groups."""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["build_optimizer", "parameter_groups"]


def parameter_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into decayed and non-decayed groups.

    Norm gains, biases and embeddings follow the usual convention: no
    weight decay is applied to 1-D parameters.
    """

    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("embedding.weight"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(
    model: Any,
    name: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> Any:
    """Build the configured optimizer."""

    groups = parameter_groups(model, weight_decay)
    key = (name or "adamw").lower()
    if key == "adamw":
        import torch

        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps)
    if key == "adam":
        import torch

        return torch.optim.Adam(groups, lr=learning_rate, betas=betas, eps=eps)
    if key == "sgd":
        import torch

        return torch.optim.SGD(groups, lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name!r}")


def optimizer_state_bytes(optimizer: Any) -> int:
    """Approximate the memory footprint of the optimizer state."""

    total = 0
    state = getattr(optimizer, "state", {}) or {}
    for entry in state.values():
        for value in entry.values():
            if hasattr(value, "numel"):
                total += int(value.numel()) * int(value.element_size())
    return total


def trainable_parameter_count(parameters: Iterable[Any]) -> int:
    """Count the parameters in an iterable."""

    return int(sum(parameter.numel() for parameter in parameters))
