"""Timing helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

__all__ = ["now_ms", "Timer", "Stopwatch"]


def now_ms() -> float:
    """Return the current time in milliseconds since the epoch."""

    return time.time() * 1000.0


class Timer:
    """Context manager measuring the wall clock duration of a block."""

    def __init__(self) -> None:
        self.started = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000.0
        return False


@dataclass
class Stopwatch:
    """Accumulates elapsed time across several measurements."""

    total_ms: float = 0.0
    count: int = 0
    started: float = field(default_factory=time.perf_counter)

    def reset(self) -> None:
        self.total_ms = 0.0
        self.count = 0
        self.started = time.perf_counter()

    def record(self, elapsed_ms: float) -> None:
        self.total_ms += elapsed_ms
        self.count += 1

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0
