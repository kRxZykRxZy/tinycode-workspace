"""Lightweight tracing of the request lifecycle."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Span", "Tracer"]


@dataclass
class Span:
    """A single traced operation."""

    name: str
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000.0

    def end(self) -> "Span":
        self.ended_at = time.perf_counter()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 3),
            "attributes": dict(self.attributes),
        }


class Tracer:
    """Collects the spans of the requests."""

    def __init__(self, keep: int = 200) -> None:
        self.spans: list[Span] = []
        self.keep = keep

    def start(self, name: str, **attributes: Any) -> Span:
        span = Span(name=name, attributes=attributes)
        self.spans.append(span)
        if len(self.spans) > self.keep:
            del self.spans[: len(self.spans) - self.keep]
        return span

    def recent(self, count: int = 20) -> list[dict[str, Any]]:
        return [span.to_dict() for span in self.spans[-count:]]
