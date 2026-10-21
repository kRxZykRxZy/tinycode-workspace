"""Background watchdog that watches resource usage."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .monitor import ResourceMonitor, ResourceSample
from .thresholds import ResourceLevel, ResourceThresholds

__all__ = ["Watchdog"]


class Watchdog:
    """Polls the resource monitor and notifies a callback under pressure.

    The watchdog never stops training by itself: it only raises events so
    that the training loop can apply the recovery policy.  This keeps the
    dashboard and the training loop independent.
    """

    def __init__(
        self,
        monitor: ResourceMonitor,
        thresholds: ResourceThresholds,
        on_pressure: Callable[[ResourceLevel, ResourceSample], None] | None = None,
        interval_seconds: float = 5.0,
    ) -> None:
        self.monitor = monitor
        self.thresholds = thresholds
        self.on_pressure = on_pressure
        self.interval_seconds = max(0.5, float(interval_seconds))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_level: ResourceLevel = ResourceLevel.NORMAL

    def start(self) -> None:
        """Start the watchdog thread."""

        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tinycode-watchdog", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the watchdog thread."""

        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self.monitor.sample()
            level = self.thresholds.classify(sample.cpu_used_ratio)
            if sample.gpu_used_ratio is not None:
                gpu_level = self.thresholds.classify(sample.gpu_used_ratio)
                if gpu_level.value != ResourceLevel.NORMAL.value:
                    level = gpu_level
            if level != self.last_level:
                self.last_level = level
                if self.on_pressure is not None:
                    try:
                        self.on_pressure(level, sample)
                    except Exception:  # pragma: no cover - callbacks must not kill the watchdog
                        pass
            self._stop.wait(self.interval_seconds)

    def status(self) -> dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "interval_seconds": self.interval_seconds,
            "last_level": self.last_level.value,
        }


def sleep_with_heartbeat(seconds: float, interval: float = 1.0) -> None:  # pragma: no cover
    """Sleep in small steps so that shutdown stays responsive."""

    deadline = time.time() + seconds
    while time.time() < deadline:
        time.sleep(min(interval, max(0.0, deadline - time.time())))
