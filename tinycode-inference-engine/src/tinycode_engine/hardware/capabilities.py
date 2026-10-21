"""Hardware capability reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..backends.interface import Backend, BackendCapabilities

__all__ = ["CapabilityReport", "report_capabilities"]


@dataclass
class CapabilityReport:
    """The capabilities of the runtime."""

    backend: str = "none"
    capabilities: BackendCapabilities | None = None
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
            "notes": list(self.notes),
        }


def report_capabilities(backend: Backend | None) -> CapabilityReport:
    """Return the capability report of a backend."""

    if backend is None:
        return CapabilityReport(backend="none", notes=["no backend is loaded"])
    capabilities = backend.capabilities()
    notes = [capabilities.notes] if capabilities.notes else []
    if capabilities.is_accelerated:
        notes.append(f"execution provider: {capabilities.execution_provider}")
    else:
        notes.append("the CPU path is in use; no accelerator was verified")
    return CapabilityReport(backend=capabilities.name, capabilities=capabilities, notes=notes)
