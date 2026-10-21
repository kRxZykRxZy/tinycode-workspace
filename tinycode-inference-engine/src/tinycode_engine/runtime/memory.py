"""Memory management for the runtime.

The engine keeps a simple accounting of the memory it uses so that the
scheduler can apply admission control and backpressure.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MemoryAccountant", "MemorySnapshot", "Arena"]


@dataclass
class MemorySnapshot:
    """A point in time view of the memory usage."""

    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_bytes: int = 0
    entries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_bytes": self.peak_bytes,
            "entries": self.entries,
            "allocated_mb": round(self.allocated_bytes / (1024**2), 3),
            "reserved_mb": round(self.reserved_bytes / (1024**2), 3),
        }


class MemoryAccountant:
    """Thread safe accounting of the runtime memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._allocated = 0
        self._reserved = 0
        self._peak = 0
        self._entries = 0

    def allocate(self, size: int) -> None:
        with self._lock:
            self._allocated += int(size)
            self._entries += 1
            self._peak = max(self._peak, self._allocated)

    def release(self, size: int) -> None:
        with self._lock:
            self._allocated = max(0, self._allocated - int(size))
            self._entries = max(0, self._entries - 1)

    def reserve(self, size: int) -> None:
        with self._lock:
            self._reserved += int(size)

    def snapshot(self) -> MemorySnapshot:
        with self._lock:
            return MemorySnapshot(
                allocated_bytes=self._allocated,
                reserved_bytes=self._reserved,
                peak_bytes=self._peak,
                entries=self._entries,
            )

    def reset(self) -> None:
        with self._lock:
            self._allocated = 0
            self._reserved = 0
            self._entries = 0


class Arena:
    """A simple bump allocator for small, short lived buffers.

    The arena keeps the bookkeeping trivial: it is used for the temporary
    buffers of the request pipeline, never for model weights.
    """

    def __init__(self, block_size: int = 1 << 20) -> None:
        self.block_size = max(4096, int(block_size))
        self._blocks: list[bytearray] = []
        self._used = 0

    def allocate(self, size: int) -> memoryview:
        """Allocate ``size`` bytes from the arena."""

        size = int(size)
        if not self._blocks or self._used + size > len(self._blocks[-1]):
            self._blocks.append(bytearray(max(self.block_size, size)))
            self._used = 0
        block = self._blocks[-1]
        view = memoryview(block)[self._used : self._used + size]
        self._used += size
        return view

    def release(self) -> None:
        """Drop every block."""

        self._blocks.clear()
        self._used = 0

    @property
    def bytes_reserved(self) -> int:
        return int(sum(len(block) for block in self._blocks))
