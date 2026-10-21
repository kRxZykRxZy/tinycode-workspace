"""Weight initialisation schemes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["initialise_weights", "build_initialisation_plan"]


def _normal_(tensor: torch.Tensor, std: float) -> None:
    nn.init.normal_(tensor, mean=0.0, std=std)


def build_initialisation_plan(model: nn.Module, d_model: int, vocab_size: int) -> dict[str, str]:
    """Return a description of the initialisation applied to a model."""

    return {
        "embedding": "normal(0, 0.02)",
        "linear": "scaled normal (0, 0.02 / sqrt(2 * n_layers))",
        "output_head": "normal(0, 0.02)",
        "norm": "ones",
        "d_model": str(d_model),
        "vocab_size": str(vocab_size),
    }


def initialise_weights(model: nn.Module, d_model: int | None = None, vocab_size: int | None = None) -> None:
    """Apply the TinyCode initialisation scheme in place.

    * embeddings and the untied output head use ``N(0, 0.02)``
    * residual projections use a depth scaled normal
    * normalisation gains start at one
    """

    if d_model is None:
        d_model = getattr(model, "config", None)
        d_model = getattr(d_model, "d_model", 0) if d_model is not None else 0
    n_layers = len(getattr(model, "layers", []) or [])
    residual_std = 0.02 / math.sqrt(max(2 * max(n_layers, 1), 1))

    for module in model.modules():
        if isinstance(module, nn.Linear):
            _normal_(module.weight, residual_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            _normal_(module.weight, 0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "weight") and module.__class__.__name__ == "RMSNorm":
            nn.init.ones_(module.weight)

    if hasattr(model, "lm_head"):
        head = model.lm_head
        inner = getattr(head, "lm_head", head)
        if isinstance(inner, nn.Linear):
            _normal_(inner.weight, 0.02)
            if inner.bias is not None:
                nn.init.zeros_(inner.bias)
