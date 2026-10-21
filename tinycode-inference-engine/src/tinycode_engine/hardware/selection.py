"""Backend selection for the inference engine."""

from __future__ import annotations

from typing import Any

from ..backends.registry import BackendSelection, BackendRegistry, default_registry

__all__ = ["select", "selection_report"]


def select(preferred: str = "auto", threads: int | None = None) -> BackendSelection:
    """Select a backend using the capability based strategy."""

    registry = default_registry({"preferred": preferred, "threads": threads})
    return registry.select()


def selection_report(selection: BackendSelection) -> dict[str, Any]:
    """Return a report of the backend selection."""

    report = selection.to_dict()
    report["attempts"] = [
        {
            "backend": attempt.get("backend"),
            "ok": attempt.get("ok"),
            "reason": attempt.get("reason"),
        }
        for attempt in report.get("attempts", [])
    ]
    return report
