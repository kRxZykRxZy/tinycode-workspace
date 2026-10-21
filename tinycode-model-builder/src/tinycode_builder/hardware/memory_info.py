"""System memory and disk detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["MemoryInfo", "detect_memory", "disk_free_bytes"]


@dataclass
class MemoryInfo:
    """Snapshot of host memory and disk availability."""

    total_ram_bytes: int = 0
    available_ram_bytes: int = 0
    used_ratio: float = 0.0
    swap_total_bytes: int = 0
    disk_total_bytes: int = 0
    disk_free_bytes: int = 0
    cpu_count: int = 1
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / (1024**3)

    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / (1024**3)

    @property
    def disk_free_gb(self) -> float:
        return self.disk_free_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ram_bytes": self.total_ram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "used_ratio": self.used_ratio,
            "swap_total_bytes": self.swap_total_bytes,
            "disk_total_bytes": self.disk_total_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "cpu_count": self.cpu_count,
            "total_ram_gb": round(self.total_ram_gb, 3),
            "available_ram_gb": round(self.available_ram_gb, 3),
            "disk_free_gb": round(self.disk_free_gb, 3),
            "details": self.details,
        }


def detect_memory(path: str | None = None) -> MemoryInfo:
    """Collect host memory, CPU count and free disk space."""

    info = MemoryInfo()
    try:
        import psutil

        virtual = psutil.virtual_memory()
        info.total_ram_bytes = int(virtual.total)
        info.available_ram_bytes = int(virtual.available)
        info.used_ratio = float(virtual.percent) / 100.0
        swap = psutil.swap_memory()
        info.swap_total_bytes = int(swap.total)
        info.cpu_count = int(psutil.cpu_count(logical=True) or 1)
    except Exception as exc:  # pragma: no cover - psutil is a hard dependency
        info.details["psutil_error"] = str(exc)
        import os

        info.cpu_count = os.cpu_count() or 1

    free, total = disk_free_bytes(path)
    info.disk_free_bytes = free
    info.disk_total_bytes = total
    return info


def disk_free_bytes(path: str | None = None) -> tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for the filesystem holding ``path``."""

    import os

    target = path or os.getcwd()
    try:
        usage = os.statvfs(target)  # type: ignore[attr-defined]
        free = usage.f_bavail * usage.f_frsize
        total = usage.f_blocks * usage.f_frsize
        return int(free), int(total)
    except (AttributeError, OSError):
        pass
    try:
        import shutil

        usage = shutil.disk_usage(target)
        return int(usage.free), int(usage.total)
    except OSError:  # pragma: no cover
        return 0, 0
