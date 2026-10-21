"""Dashboard controls.

The controls never execute shell commands.  They only set flags that the
training loop observes, which keeps the training run alive even when the
dashboard or the tunnel disconnects.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .state import DashboardState

__all__ = ["ControlResult", "ControlBridge"]


@dataclass
class ControlResult:
    """Outcome of a control action."""

    ok: bool = False
    action: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "message": self.message,
            "details": dict(self.details),
        }


class ControlBridge:
    """Connects the HTTP layer with the training loop."""

    def __init__(
        self,
        state: DashboardState,
        checkpoint_callback: Callable[[], Any] | None = None,
        pause_callback: Callable[[], None] | None = None,
        resume_callback: Callable[[], None] | None = None,
        stop_callback: Callable[[], None] | None = None,
    ) -> None:
        self.state = state
        self._checkpoint = checkpoint_callback
        self._pause = pause_callback
        self._resume = resume_callback
        self._stop = stop_callback
        self._lock = threading.Lock()
        self.last_actions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def request_checkpoint(self) -> ControlResult:
        self.state.request_checkpoint()
        details: dict[str, Any] = {}
        if self._checkpoint is not None:
            try:
                info = self._checkpoint()
                details = info.to_dict() if hasattr(info, "to_dict") else {"result": str(info)}
            except Exception as exc:
                return ControlResult(False, "checkpoint", f"Checkpoint failed: {exc}", details)
        result = ControlResult(True, "checkpoint", "Checkpoint requested", details)
        self._record(result)
        return result

    def request_pause(self) -> ControlResult:
        self.state.request_pause()
        if self._pause is not None:
            try:
                self._pause()
            except Exception as exc:  # pragma: no cover - defensive
                return ControlResult(False, "pause", f"Pause failed: {exc}")
        result = ControlResult(True, "pause", "Pause requested")
        self._record(result)
        return result

    def request_resume(self) -> ControlResult:
        self.state.request_resume()
        if self._resume is not None:
            try:
                self._resume()
            except Exception as exc:  # pragma: no cover - defensive
                return ControlResult(False, "resume", f"Resume failed: {exc}")
        result = ControlResult(True, "resume", "Resume requested")
        self._record(result)
        return result

    def request_stop(self) -> ControlResult:
        self.state.request_stop()
        if self._stop is not None:
            try:
                self._stop()
            except Exception as exc:  # pragma: no cover - defensive
                return ControlResult(False, "stop", f"Stop failed: {exc}")
        result = ControlResult(True, "stop", "Graceful stop requested")
        self._record(result)
        return result

    def _record(self, result: ControlResult) -> None:
        with self._lock:
            self.last_actions.append({"timestamp": time.time(), **result.to_dict()})
            if len(self.last_actions) > 100:
                del self.last_actions[:50]
