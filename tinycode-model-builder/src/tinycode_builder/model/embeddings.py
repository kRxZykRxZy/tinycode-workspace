"""Token and position embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .architecture import TransformerConfig

__all__ = ["TokenEmbedding", "PositionalEncoding", "build_embeddings"]


class TokenEmbedding(nn.Module):
    """Token embedding table plus optional learned position embeddings."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.d_model)
        self.position: nn.Module | None = None
        if config.position_encoding == "learned":
            self.position = nn.Embedding(config.context_length, config.d_model)
        elif config.position_encoding != "sinusoidal":
            raise ValueError(f"Unknown position encoding: {config.position_encoding!r}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token(input_ids)
        if self.position is not None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            hidden = hidden + self.position(positions).unsqueeze(0)
        else:
            hidden = hidden + sinusoidal_positions(
                input_ids.shape[1], self.config.d_model, device=hidden.device, dtype=hidden.dtype
            ).unsqueeze(0)
        return hidden

    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return f"vocab={self.config.vocab_size}, dim={self.config.d_model}"


def sinusoidal_positions(
    length: int, dim: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return sinusoidal position embeddings of shape ``(length, dim)``."""

    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div_term = torch.exp(-math.log(10000.0) * index / dim)
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype)


def build_embeddings(config: TransformerConfig) -> TokenEmbedding:
    """Instantiate the embedding stack described by ``config``."""

    return TokenEmbedding(config)
