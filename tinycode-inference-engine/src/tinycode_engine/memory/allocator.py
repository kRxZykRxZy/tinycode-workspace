"""Buffer allocation for the runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from .arena import Arena

__all__ = ["BufferAllocator"]


class BufferAllocator:
    """Allocates the temporary buffers of the request pipeline."""

    def __init__(self, block_size: int = 1 << 20) -> None:
        self.arena = Arena(block_size=block_size)

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float32) -> Any:
        """Allocate a zeroed buffer."""

        return np.zeros(shape, dtype=dtype)

    def scratch(self, size: int) -> memoryview:
        """Allocate a raw scratch buffer."""

        return self.arena.allocate(size)

    def release(self) -> None:
        """Release every buffer."""

        self.arena.release()

    @property
    def bytes_reserved(self) -> int:
        return self.arena.bytes_reserved
