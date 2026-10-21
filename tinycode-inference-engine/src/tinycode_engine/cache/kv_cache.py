"""The key/value cache.

The cache is bounded by the fixed context length (1024 tokens).  It is
never allocated without an upper bound and it is released as soon as a
request finishes or is cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["KVCache", "CacheStats"]


@dataclass
class CacheStats:
    """Memory accounting for a cache."""

    allocated_bytes: int = 0
    used_positions: int = 0
    context_length: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "used_positions": self.used_positions,
            "context_length": self.context_length,
            "utilisation": (
                self.used_positions / self.context_length if self.context_length else 0.0
            ),
        }


class KVCache:
    """A per-request KV cache."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        context_length: int = 1024,
        dtype: Any = np.float32,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.context_length = int(context_length)
        self.dtype = dtype
        shape = (n_heads, self.context_length, head_dim)
        self.keys = [np.zeros(shape, dtype=dtype) for _ in range(n_layers)]
        self.values = [np.zeros(shape, dtype=dtype) for _ in range(n_layers)]
        self.length = 0

    # ------------------------------------------------------------------
    @property
    def allocated_bytes(self) -> int:
        return int(sum(key.nbytes + value.nbytes for key, value in zip(self.keys, self.values)))

    @property
    def remaining(self) -> int:
        return max(0, self.context_length - self.length)

    def full(self) -> bool:
        return self.length >= self.context_length

    def can_append(self, count: int) -> bool:
        return self.length + count <= self.context_length

    def append(self, layer: int, key: Any, value: Any) -> int:
        """Store the key/value vectors of one position.

        Returns:
            The number of positions stored after the append.
        """

        if layer < 0 or layer >= self.n_layers:
            raise IndexError(f"Invalid layer index: {layer}")
        if self.full():
            raise RuntimeError(
                f"KV cache is full: the fixed {self.context_length} token context is exhausted."
            )
        position = self.length
        self.keys[layer][:, position, :] = key[:, 0, :]
        self.values[layer][:, position, :] = value[:, 0, :]
        return position + 1

    def advance(self, count: int) -> None:
        """Advance the write position after a batch of positions was stored."""

        if self.length + count > self.context_length:
            raise RuntimeError(
                f"KV cache overflow: {self.length + count} > {self.context_length} tokens."
            )
        self.length += count

    def reset(self) -> None:
        """Reset the cache so that it can be reused."""

        self.length = 0

    def release(self) -> None:
        """Free the memory held by the cache."""

        self.keys = []
        self.values = []
        self.length = 0

    def stats(self) -> CacheStats:
        return CacheStats(
            allocated_bytes=self.allocated_bytes,
            used_positions=self.length,
            context_length=self.context_length,
        )
