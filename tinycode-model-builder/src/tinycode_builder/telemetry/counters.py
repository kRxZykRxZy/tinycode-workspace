"""Metric counters used by the dashboard and the reports."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Counters", "Timer"]


@dataclass
class Counters:
    """Thread safe counters and gauges."""

    counters: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment(self, name: str, amount: float = 1.0) -> float:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + amount
            return self.counters[name]

    def set(self, name: str, value: float) -> float:
        with self._lock:
            self.gauges[name] = float(value)
            return self.gauges[name]

    def get(self, name: str, default: float = 0.0) -> float:
        with self._lock:
            return float(self.counters.get(name, self.gauges.get(name, default)))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
            }

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.gauges.clear()


class Timer:
    """Context manager that measures the duration of a block."""

    def __init__(self) -> None:
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = time.perf_counter() - self.started
        return False
