"""The TinyCode causal language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration.immutable_constraints import CONTEXT_LENGTH, assert_context_length
from .architecture import TransformerConfig
from .attention import TransformerBlock
from .embeddings import TokenEmbedding
from .initialization import initialise_weights
from .normalization import build_norm
from .output_head import OutputHead

__all__ = ["TinyCodeCausalLM", "ModelOutput", "build_model"]


@dataclass
class ModelOutput:
    """Container for the forward pass results."""

    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None
    loss: torch.Tensor | None = None

    @property
    def next_token_logits(self) -> torch.Tensor:
        return self.logits[:, -1, :]


class TinyCodeCausalLM(nn.Module):
    """Decoder-only causal language model with a fixed 1024 token context."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert_context_length(config.context_length)
        self.config = config
        self.embeddings = TokenEmbedding(config)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = build_norm(config.norm_type, config.d_model, eps=config.norm_eps)
        self.lm_head = OutputHead(config)
        initialise_weights(self, config.d_model, config.vocab_size)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> ModelOutput:
        """Run the model.

        Args:
            input_ids: ``(batch, seq)`` token ids with ``seq <= 1024``.
            labels: optional targets for the language modelling loss.
            return_hidden: also return the final hidden states.
        """

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2-D, got shape {tuple(input_ids.shape)}")
        if input_ids.shape[1] > CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds the fixed "
                f"{CONTEXT_LENGTH} token context."
            )

        hidden = self.embeddings(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden, self.embeddings.token.weight)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return ModelOutput(
            logits=logits, hidden_states=hidden if return_hidden else None, loss=loss
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the parameters of the model."""

        total = 0
        for parameter in self.parameters():
            if trainable_only and not parameter.requires_grad:
                continue
            total += parameter.numel()
        return int(total)

    def config_dict(self) -> dict[str, Any]:
        """Return the architecture description used by the artifact manifest."""

        return self.config.to_dict()

    def save_pretrained(self, directory: str) -> str:
        """Save the raw weights to ``directory`` in a portable format."""

        from pathlib import Path

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target / "model.pt")
        return str(target)


def build_model(config: TransformerConfig) -> TinyCodeCausalLM:
    """Instantiate the model described by ``config``."""

    return TinyCodeCausalLM(config)
