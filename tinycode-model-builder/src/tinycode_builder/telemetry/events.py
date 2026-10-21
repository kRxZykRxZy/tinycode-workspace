"""Structured events for the builder."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = ["Event", "EventBus"]


@dataclass
class Event:
    """A timestamped builder event."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    level: str = "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "level": self.level,
        }

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


class EventBus:
    """Collects events, notifies listeners and optionally writes JSONL."""

    def __init__(self, log_path: str | Path | None = None, keep_in_memory: int = 500) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.keep_in_memory = keep_in_memory
        self.events: list[Event] = []
        self.listeners: list[Callable[[Event], None]] = []

    def subscribe(self, listener: Callable[[Event], None]) -> None:
        self.listeners.append(listener)

    def emit(self, name: str, payload: dict[str, Any] | None = None, level: str = "info") -> Event:
        event = Event(name=name, payload=payload or {}, level=level)
        self.events.append(event)
        if len(self.events) > self.keep_in_memory:
            del self.events[: len(self.events) - self.keep_in_memory]
        if self.log_path is not None:
            self._write(event)
        for listener in list(self.listeners):
            try:
                listener(event)
            except Exception:  # pragma: no cover - listeners must not break the emitter
                continue
        return event

    def _write(self, event: Event) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_line() + "\n")
        except OSError:  # pragma: no cover
            pass

    def recent(self, count: int = 50) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events[-count:]]

    def clear(self) -> None:
        self.events.clear()
