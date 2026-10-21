"""A worker of the inference server.

A worker owns a slice of the request queue.  The default deployment uses
a single worker because the model execution is not thread safe for
concurrent forward passes of the same weights.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Worker", "WorkerPool", "WorkerStats"]


@dataclass
class WorkerStats:
    """Counters of a worker."""

    processed: int = 0
    failed: int = 0
    cancelled: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "failed": self.failed,
            "cancelled": self.cancelled,
        }


@dataclass
class Worker:
    """Executes the queued requests of one worker."""

    worker_id: int = 0
    engine: Any = None
    stats: WorkerStats = field(default_factory=WorkerStats)
    _thread: threading.Thread | None = None
    _stop = threading.Event()

    def start(self) -> None:
        """Start the worker thread."""

        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"tinycode-worker-{self.worker_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker thread."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        scheduler = getattr(self.engine, "scheduler", None)
        if scheduler is None:
            return
        while not self._stop.is_set():
            scheduler.run(max_rounds=16)
            if not scheduler.queue.peek() and not scheduler.active:
                self._stop.wait(0.05)
