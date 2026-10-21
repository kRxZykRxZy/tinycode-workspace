"""Request level continuous batching.

The scheduler keeps several requests active and advances them round
robin, one decoding step per request and round.  Every request owns its
own KV cache and its own state; a cancelled or finished request releases
its resources immediately and never affects the others.

The batching is *request level*: the TinyCode decoder is executed with a
batch size of one because the architecture does not implement a batched
attention path.  The scheduler therefore interleaves requests instead of
batching their tensors.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..budget.controller import RequestBudget
from ..generation.generator import GenerationSession, TokenChunk
from .admission import AdmissionController
from .cancellation import CancellationRegistry
from .priorities import Priority
from .request_queue import RequestQueue

__all__ = ["ActiveRequest", "ContinuousBatchingScheduler", "SchedulerStats"]


@dataclass
class ActiveRequest:
    """A request that is currently being served."""

    request_id: str
    session: GenerationSession
    budget: RequestBudget | None = None
    priority: Priority = Priority.NORMAL
    started_at: float = field(default_factory=time.time)
    finished: bool = False
    finish_reason: str | None = None
    error: str | None = None
    on_token: Callable[[TokenChunk], None] | None = None

    @property
    def tokens_generated(self) -> int:
        return len(self.session.generated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tokens_generated": self.tokens_generated,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "priority": self.priority.name.lower(),
            "elapsed_ms": round((time.time() - self.started_at) * 1000.0, 3),
            "error": self.error,
        }


@dataclass
class SchedulerStats:
    """Counters of the scheduler."""

    admitted: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    rejected: int = 0
    tokens_generated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "failed": self.failed,
            "rejected": self.rejected,
            "tokens_generated": self.tokens_generated,
        }


class ContinuousBatchingScheduler:
    """A scheduler that interleaves the decoding of several requests."""

    def __init__(
        self,
        session_factory: Callable[[str, Any], GenerationSession],
        max_active: int = 8,
        queue_capacity: int = 64,
        admission: AdmissionController | None = None,
        cancellations: CancellationRegistry | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.max_active = max(1, max_active)
        self.queue: RequestQueue = RequestQueue(capacity=queue_capacity)
        self.admission = admission or AdmissionController(queue=self.queue, max_active_requests=self.max_active)
        self.cancellations = cancellations or CancellationRegistry()
        self.active: dict[str, ActiveRequest] = {}
        self._pending_meta: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.stats = SchedulerStats()

    # ------------------------------------------------------------------
    def submit(
        self,
        request_id: str,
        payload: Any,
        priority: Priority = Priority.NORMAL,
        budget: RequestBudget | None = None,
        on_token: Callable[[TokenChunk], None] | None = None,
    ) -> bool:
        """Submit a request to the scheduler."""

        decision = self.admission.check()
        if not decision.admitted:
            self.stats.rejected += 1
            return False
        accepted = self.queue.put(request_id, payload, priority=priority)
        if not accepted:
            self.stats.rejected += 1
            return False
        self._pending_meta[request_id] = {"budget": budget, "on_token": on_token}
        return True

    # ------------------------------------------------------------------
    def cancel(self, request_id: str) -> bool:
        """Cancel a queued or running request."""

        cancelled = self.cancellations.cancel(request_id)
        queued = self.queue.cancel(request_id)
        with self._lock:
            active = self.active.get(request_id)
        if active is not None:
            active.session.finished = True
            active.session.finish_reason = "cancelled"
            active.finished = True
            active.finish_reason = "cancelled"
        if active is not None or queued:
            self.stats.cancelled += 1
        return cancelled or queued

    # ------------------------------------------------------------------
    def run(self, max_rounds: int | None = None) -> None:
        """Run the scheduler loop until every request is finished."""

        rounds = 0
        while True:
            self._admit_pending()
            if not self.active and self.queue.peek() is None:
                break
            self._advance_active()
            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

    def _admit_pending(self) -> None:
        with self._lock:
            free_slots = self.max_active - len(self.active)
        while free_slots > 0:
            item = self.queue.get()
            if item is None:
                break
            meta = self._pending_meta.pop(item.request_id, {})
            try:
                session = self.session_factory(item.request_id, item.payload)
            except Exception as exc:
                self.stats.failed += 1
                continue
            with self._lock:
                self.active[item.request_id] = ActiveRequest(
                    request_id=item.request_id,
                    session=session,
                    budget=meta.get("budget"),
                    on_token=meta.get("on_token"),
                )
            self.admission.on_start()
            self.stats.admitted += 1
            free_slots -= 1

    def _advance_active(self) -> None:
        with self._lock:
            active = list(self.active.values())
        for request in active:
            if request.finished:
                continue
            if self.cancellations.is_cancelled(request.request_id):
                request.session.finished = True
                request.session.finish_reason = "cancelled"
                request.finished = True
                request.finish_reason = "cancelled"
                self.stats.cancelled += 1
                continue
            try:
                chunk = request.session.step()
            except Exception as exc:
                request.error = str(exc)
                request.finished = True
                request.finish_reason = "error"
                self.stats.failed += 1
                continue

            if chunk is not None and request.on_token is not None:
                try:
                    request.on_token(chunk)
                except Exception:  # pragma: no cover - listeners must not break the loop
                    pass
            self.stats.tokens_generated += 1

            if request.session.finished:
                request.finished = True
                request.finish_reason = request.session.finish_reason
                self.stats.completed += 1
                request.session.release()

        with self._lock:
            finished = [key for key, value in self.active.items() if value.finished]
            for key in finished:
                self.active.pop(key, None)
                self.admission.on_finish()
                self.cancellations.release(key)

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the scheduler state."""

        with self._lock:
            active = [request.to_dict() for request in self.active.values()]
        return {
            "active": active,
            "active_count": len(active),
            "max_active": self.max_active,
            "queue": self.queue.stats(),
            "stats": self.stats.to_dict(),
        }
