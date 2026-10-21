"""Positional encoding for the numpy runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["position_embeddings", "sinusoidal_positions"]


def sinusoidal_positions(length: int, dim: int) -> Any:
    """Sinusoidal position embeddings of shape ``(length, dim)``."""

    position = np.arange(length, dtype=np.float32)[:, None]
    index = np.arange(0, dim, 2, dtype=np.float32)
    div_term = np.exp(-np.log(10000.0) * index / dim)
    encoding = np.zeros((length, dim), dtype=np.float32)
    encoding[:, 0::2] = np.sin(position * div_term)
    encoding[:, 1::2] = np.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding


def position_embeddings(
    position_encoding: str,
    length: int,
    dim: int,
    learned_weight: Any = None,
) -> Any:
    """Return the position embeddings for ``length`` positions."""

    if position_encoding == "learned":
        if learned_weight is None:
            raise ValueError("Learned positional embeddings require the weight matrix.")
        return learned_weight[:length]
    if position_encoding == "sinusoidal":
        return sinusoidal_positions(length, dim)
    raise ValueError(f"Unsupported position encoding: {position_encoding!r}")
