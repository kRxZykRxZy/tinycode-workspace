"""Cancellation of running and queued requests."""

from __future__ import annotations

import threading
from typing import Any, Callable

__all__ = ["CancellationRegistry"]


class CancellationRegistry:
    """Tracks the cancellation state of the requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._callbacks: dict[str, list[Callable[[], None]]] = {}

    # ------------------------------------------------------------------
    def cancel(self, request_id: str) -> bool:
        """Request the cancellation of a request."""

        with self._lock:
            already = request_id in self._cancelled
            self._cancelled.add(request_id)
            callbacks = list(self._callbacks.get(request_id, []))
        for callback in callbacks:
            try:
                callback()
            except Exception:  # pragma: no cover - callbacks must not raise
                continue
        return not already

    def is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancelled

    def register(self, request_id: str, callback: Callable[[], None]) -> None:
        """Register a callback that is invoked when the request is cancelled."""

        with self._lock:
            self._callbacks.setdefault(request_id, []).append(callback)
            already_cancelled = request_id in self._cancelled
        if already_cancelled:
            try:
                callback()
            except Exception:  # pragma: no cover
                pass

    def release(self, request_id: str) -> None:
        """Forget a request (it has finished)."""

        with self._lock:
            self._cancelled.discard(request_id)
            self._callbacks.pop(request_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cancelled.clear()
            self._callbacks.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"cancelled": len(self._cancelled)}
