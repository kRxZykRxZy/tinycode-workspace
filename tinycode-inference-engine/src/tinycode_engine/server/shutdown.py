"""Graceful shutdown of the inference server."""

from __future__ import annotations

import time
from typing import Any

__all__ = ["ShutdownReport", "shutdown_engine"]


class ShutdownReport:
    """What the shutdown sequence did."""

    def __init__(self) -> None:
        self.stopped_requests = 0
        self.errors: list[str] = []
        self.elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stopped_requests": self.stopped_requests,
            "errors": list(self.errors),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


def shutdown_engine(engine: Any, timeout: float = 10.0) -> ShutdownReport:
    """Stop the running requests and release the resources."""

    report = ShutdownReport()
    started = time.perf_counter()

    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None:
        for request_id in list(scheduler.active):
            try:
                scheduler.cancel(request_id)
                report.stopped_requests += 1
            except Exception as exc:  # pragma: no cover
                report.errors.append(f"{request_id}: {exc}")

    try:
        engine.shutdown()
    except Exception as exc:  # pragma: no cover
        report.errors.append(str(exc))

    report.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return report
