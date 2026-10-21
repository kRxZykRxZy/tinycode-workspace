"""Runtime resource monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ResourceSample", "ResourceMonitor"]


@dataclass
class ResourceSample:
    """A single measurement of the machine's resource usage."""

    timestamp: float
    cpu_used_ratio: float = 0.0
    cpu_available_bytes: int = 0
    gpu_used_ratio: float | None = None
    gpu_used_bytes: int | None = None
    gpu_total_bytes: int | None = None
    disk_free_bytes: int = 0
    gpu_temperature_c: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_used_ratio": round(self.cpu_used_ratio, 4),
            "cpu_available_bytes": self.cpu_available_bytes,
            "gpu_used_ratio": None if self.gpu_used_ratio is None else round(self.gpu_used_ratio, 4),
            "gpu_used_bytes": self.gpu_used_bytes,
            "gpu_total_bytes": self.gpu_total_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "gpu_temperature_c": self.gpu_temperature_c,
        }


@dataclass
class ResourceMonitor:
    """Samples host and GPU resource usage."""

    history_size: int = 600
    samples: list[ResourceSample] = field(default_factory=list)

    def sample(self) -> ResourceSample:
        """Take a single measurement."""

        sample = ResourceSample(timestamp=time.time())
        try:
            import psutil

            virtual = psutil.virtual_memory()
            sample.cpu_used_ratio = float(virtual.percent) / 100.0
            sample.cpu_available_bytes = int(virtual.available)
            disk = psutil.disk_usage(str(_safe_cwd()))
            sample.disk_free_bytes = int(disk.free)
        except Exception:  # pragma: no cover - psutil optional at runtime
            pass

        try:
            import torch

            if torch.cuda.is_available():  # pragma: no cover - hardware dependent
                free, total = torch.cuda.mem_get_info()
                sample.gpu_used_bytes = int(total - free)
                sample.gpu_total_bytes = int(total)
                if total > 0:
                    sample.gpu_used_ratio = (total - free) / total
                try:
                    sample.gpu_temperature_c = float(
                        torch.cuda.temperature(torch.cuda.current_device())
                    )
                except Exception:
                    sample.gpu_temperature_c = None
        except Exception:
            pass

        self.samples.append(sample)
        if len(self.samples) > self.history_size:
            del self.samples[: len(self.samples) - self.history_size]
        return sample

    def latest(self) -> ResourceSample | None:
        return self.samples[-1] if self.samples else None

    def to_dict(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "latest": latest.to_dict() if latest else None,
            "samples": len(self.samples),
        }


def _safe_cwd() -> str:
    import os

    return os.getcwd()
