"""Backpressure.

When the server is saturated the engine stops admitting work and signals
the pressure so that the API can answer with a retry hint instead of
queueing without a bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BackpressureSignal", "BackpressureController"]


@dataclass
class BackpressureSignal:
    """The current pressure of the server."""

    active: bool = False
    level: float = 0.0
    reason: str = ""
    retry_after_seconds: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "level": round(self.level, 4),
            "reason": self.reason,
            "retry_after_seconds": self.retry_after_seconds,
        }


@dataclass
class BackpressureController:
    """Computes the backpressure signal from the scheduler state."""

    queue_depth: int = 0
    queue_capacity: int = 64
    active_requests: int = 0
    max_active_requests: int = 8
    memory_pressure: float = 0.0
    threshold: float = 0.8
    history: list[dict[str, Any]] = field(default_factory=list)

    def evaluate(self) -> BackpressureSignal:
        """Return the current backpressure signal."""

        queue_ratio = self.queue_depth / self.queue_capacity if self.queue_capacity else 0.0
        concurrency_ratio = (
            self.active_requests / self.max_active_requests if self.max_active_requests else 0.0
        )
        level = max(queue_ratio, concurrency_ratio, self.memory_pressure)
        if level < self.threshold:
            signal = BackpressureSignal(active=False, level=level, reason="")
        else:
            reason = "the server is saturated"
            if queue_ratio >= concurrency_ratio and queue_ratio >= self.memory_pressure:
                reason = "the request queue is nearly full"
            elif self.memory_pressure >= max(queue_ratio, concurrency_ratio):
                reason = "the server is under memory pressure"
            else:
                reason = "the server is at its concurrency limit"
            signal = BackpressureSignal(
                active=True,
                level=level,
                reason=reason,
                retry_after_seconds=round(min(5.0, max(0.25, (1.0 - level) * 5.0)), 3),
            )
        self.history.append(signal.to_dict())
        if len(self.history) > 200:
            del self.history[:100]
        return signal

    def stats(self) -> dict[str, Any]:
        return {
            "queue_depth": self.queue_depth,
            "queue_capacity": self.queue_capacity,
            "active_requests": self.active_requests,
            "max_active_requests": self.max_active_requests,
            "memory_pressure": self.memory_pressure,
        }
