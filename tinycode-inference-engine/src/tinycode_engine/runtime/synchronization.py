"""Synchronisation primitives.

The CPU backend executes synchronously, so the primitives are simple
wrappers.  Accelerated backends use them to make their behaviour
explicit instead of relying on implicit ordering.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SynchronizationContext", "CountingEvent"]


@dataclass
class CountingEvent:
    """A simple event with a counter, used to coordinate workers."""

    event: threading.Event = field(default_factory=threading.Event)
    count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self) -> None:
        self.event.set()

    def clear(self) -> None:
        self.event.clear()

    def wait(self, timeout: float | None = None) -> bool:
        return self.event.wait(timeout)

    def bump(self) -> int:
        with self._lock:
            self.count += 1
            return self.count


class SynchronizationContext:
    """Context manager that synchronises a backend on exit."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend
        self._lock = threading.Lock()

    def __enter__(self) -> "SynchronizationContext":
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self.backend is not None:
                self.backend.synchronize()
        finally:
            self._lock.release()
        return False
