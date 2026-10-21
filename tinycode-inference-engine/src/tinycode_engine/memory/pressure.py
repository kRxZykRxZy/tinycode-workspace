"""Memory pressure detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MemoryPressure", "detect_memory_pressure"]


@dataclass
class MemoryPressure:
    """The memory pressure of the machine."""

    level: float = 0.0
    available_bytes: int | None = None
    total_bytes: int | None = None
    source: str = "unavailable"

    @property
    def under_pressure(self) -> bool:
        return self.level >= 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": round(self.level, 4),
            "available_bytes": self.available_bytes,
            "total_bytes": self.total_bytes,
            "source": self.source,
            "under_pressure": self.under_pressure,
        }


def detect_memory_pressure() -> MemoryPressure:
    """Detect the memory pressure of the machine."""

    try:
        import psutil

        virtual = psutil.virtual_memory()
        total = int(virtual.total)
        available = int(virtual.available)
        level = 1.0 - (available / total) if total else 0.0
        return MemoryPressure(
            level=level, available_bytes=available, total_bytes=total, source="psutil"
        )
    except Exception:  # pragma: no cover
        return MemoryPressure()
