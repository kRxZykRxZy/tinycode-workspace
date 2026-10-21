"""A pool of inference workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .worker import Worker, WorkerStats

__all__ = ["WorkerPool"]


@dataclass
class WorkerPool:
    """Starts and stops the workers of the server."""

    engine: Any = None
    size: int = 1
    workers: list[Worker] = field(default_factory=list)

    def start(self) -> list[Worker]:
        """Start the workers."""

        self.workers = []
        for index in range(max(1, self.size)):
            worker = Worker(worker_id=index, engine=self.engine)
            worker.start()
            self.workers.append(worker)
        return self.workers

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the workers."""

        for worker in self.workers:
            worker.stop(timeout=timeout)
        self.workers = []

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self.workers),
            "workers": [worker.stats.to_dict() for worker in self.workers],
        }
