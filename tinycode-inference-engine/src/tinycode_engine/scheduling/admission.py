"""Admission control.

A request is admitted only when the queue, the memory and the worker
pool have capacity.  Requests that do not fit are deferred or rejected
with a clear error instead of degrading the whole service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..runtime.memory import MemoryAccountant
from .request_queue import RequestQueue

__all__ = ["AdmissionDecision", "AdmissionController"]


@dataclass
class AdmissionDecision:
    """The result of an admission check."""

    admitted: bool = False
    reason: str = ""
    deferred: bool = False
    status_code: int = 200

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "deferred": self.deferred,
            "status_code": self.status_code,
        }


class AdmissionController:
    """Decides whether a request may enter the scheduler."""

    def __init__(
        self,
        queue: RequestQueue | None = None,
        memory: MemoryAccountant | None = None,
        max_active_requests: int = 8,
        memory_limit_bytes: int | None = None,
    ) -> None:
        self.queue = queue
        self.memory = memory or MemoryAccountant()
        self.max_active_requests = max_active_requests
        self.memory_limit_bytes = memory_limit_bytes
        self.active_requests = 0
        self.admitted = 0
        self.rejected = 0

    # ------------------------------------------------------------------
    def check(
        self,
        estimated_memory_bytes: int = 0,
        priority: Any = None,
    ) -> AdmissionDecision:
        """Check whether a request may be admitted."""

        if self.active_requests >= self.max_active_requests:
            return AdmissionDecision(
                admitted=False,
                reason="the server is at its concurrency limit",
                deferred=True,
                status_code=503,
            )

        if self.queue is not None and len(self.queue) >= self.queue.capacity:
            return AdmissionDecision(
                admitted=False,
                reason="the request queue is full",
                deferred=True,
                status_code=503,
            )

        if self.memory_limit_bytes:
            snapshot = self.memory.snapshot()
            if snapshot.allocated_bytes + estimated_memory_bytes > self.memory_limit_bytes:
                return AdmissionDecision(
                    admitted=False,
                    reason="the server is under memory pressure",
                    deferred=True,
                    status_code=503,
                )

        self.admitted += 1
        return AdmissionDecision(admitted=True, reason="admitted")

    def on_start(self) -> None:
        self.active_requests += 1

    def on_finish(self) -> None:
        self.active_requests = max(0, self.active_requests - 1)

    def stats(self) -> dict[str, Any]:
        return {
            "active_requests": self.active_requests,
            "max_active_requests": self.max_active_requests,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "memory": self.memory.snapshot().to_dict(),
        }
