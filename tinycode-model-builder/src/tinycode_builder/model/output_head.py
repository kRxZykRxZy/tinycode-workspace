"""Output projection head.

When ``tie_embeddings`` is set the head re-uses the token embedding matrix
and therefore owns no parameters of its own.  The embedding is *not*
registered as a child module of the head: that keeps the exported
``state_dict`` free of duplicated tensors, so the number of stored
parameters always equals the number of trainable parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import TransformerConfig

__all__ = ["OutputHead", "build_output_head"]


class OutputHead(nn.Module):
    """Language modelling head."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.lm_head: nn.Linear | None = None
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(
                config.d_model, config.vocab_size, bias=config.head_bias
            )

    def forward(self, hidden: torch.Tensor, embedding_weight: torch.Tensor | None = None) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(hidden)
        if embedding_weight is None:  # pragma: no cover - defensive
            raise ValueError("A tied output head requires the token embedding weight.")
        return F.linear(hidden, embedding_weight)


def build_output_head(config: TransformerConfig) -> OutputHead:
    """Instantiate the output head."""

    return OutputHead(config)
