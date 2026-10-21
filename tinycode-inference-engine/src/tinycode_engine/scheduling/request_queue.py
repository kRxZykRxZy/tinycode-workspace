"""The bounded request queue."""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from .priorities import Priority, PriorityClass, compare_priorities

__all__ = ["QueuedRequest", "RequestQueue"]

T = TypeVar("T")


@dataclass
class QueuedRequest(Generic[T]):
    """A request that waits for admission."""

    request_id: str
    payload: T
    priority_class: PriorityClass = field(default_factory=PriorityClass)
    enqueued_at: float = field(default_factory=time.time)
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "priority": self.priority_class.to_dict(),
            "enqueued_at": self.enqueued_at,
            "cancelled": self.cancelled,
        }


class RequestQueue(Generic[T]):
    """A bounded, priority ordered queue of requests."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = max(1, int(capacity))
        self._lock = threading.Lock()
        self._heap: list[tuple[int, int, QueuedRequest[T]]] = []
        self._counter = itertools.count()
        self._rejected = 0

    # ------------------------------------------------------------------
    def put(
        self,
        request_id: str,
        payload: T,
        priority: Priority = Priority.NORMAL,
        deadline: float | None = None,
    ) -> bool:
        """Enqueue a request.

        Returns:
            ``True`` when the request was accepted, ``False`` when the
            queue is full.
        """

        with self._lock:
            if len(self._heap) >= self.capacity:
                self._rejected += 1
                return False
            priority_class = PriorityClass(
                priority=priority, deadline=deadline, enqueued_at=time.time()
            )
            item = QueuedRequest(
                request_id=request_id,
                payload=payload,
                priority_class=priority_class,
            )
            heapq.heappush(self._heap, (priority_class.value, next(self._counter), item))
            return True

    def get(self) -> QueuedRequest[T] | None:
        """Pop the next request, skipping cancelled ones."""

        with self._lock:
            while self._heap:
                _priority, _order, item = heapq.heappop(self._heap)
                if item.cancelled:
                    continue
                return item
        return None

    def peek(self) -> QueuedRequest[T] | None:
        with self._lock:
            while self._heap:
                _priority, _order, item = self._heap[0]
                if item.cancelled:
                    heapq.heappop(self._heap)
                    continue
                return item
        return None

    def cancel(self, request_id: str) -> bool:
        """Mark a queued request as cancelled."""

        with self._lock:
            for _priority, _order, item in self._heap:
                if item.request_id == request_id:
                    item.cancelled = True
                    return True
        return False

    def clear(self) -> None:
        with self._lock:
            self._heap.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def rejected(self) -> int:
        return self._rejected

    def stats(self) -> dict[str, Any]:
        with self._lock:
            depth = len(self._heap)
        return {
            "depth": depth,
            "capacity": self.capacity,
            "rejected": self._rejected,
            "utilisation": round(depth / self.capacity, 4) if self.capacity else 0.0,
        }
