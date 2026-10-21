"""The server lifecycle."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

__all__ = ["ServerLifecycle", "ServerState"]


class ServerState:
    """The lifecycle states of the server."""

    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ServerLifecycle:
    """Tracks the state of the server and the shutdown event."""

    state: str = ServerState.CREATED
    error: str | None = None
    _stop_event: threading.Event = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def mark_starting(self) -> None:
        self.state = ServerState.STARTING

    def mark_ready(self) -> None:
        self.state = ServerState.READY

    def mark_failed(self, error: str) -> None:
        self.state = ServerState.FAILED
        self.error = error

    def mark_stopping(self) -> None:
        self.state = ServerState.STOPPING

    def mark_stopped(self) -> None:
        self.state = ServerState.STOPPED
        self._stop_event.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        return self._stop_event.wait(timeout)

    def request_stop(self) -> None:
        self._stop_event.set()

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "error": self.error}
