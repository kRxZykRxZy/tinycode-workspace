"""Hardware telemetry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["HardwareSample", "HardwareTelemetry"]


@dataclass
class HardwareSample:
    """A measurement of the machine."""

    timestamp: float = field(default_factory=time.time)
    cpu_used_ratio: float | None = None
    memory_used_ratio: float | None = None
    gpu_used_ratio: float | None = None
    gpu_memory_used: int | None = None
    gpu_memory_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_used_ratio": self.cpu_used_ratio,
            "memory_used_ratio": self.memory_used_ratio,
            "gpu_used_ratio": self.gpu_used_ratio,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_total": self.gpu_memory_total,
        }


class HardwareTelemetry:
    """Samples the hardware usage of the machine."""

    def __init__(self, keep: int = 300) -> None:
        self.keep = keep
        self.samples: list[HardwareSample] = []
        self._lock = threading.Lock()

    def sample(self) -> HardwareSample:
        """Take a single measurement."""

        sample = HardwareSample()
        try:
            import psutil

            sample.cpu_used_ratio = float(psutil.cpu_percent(interval=None)) / 100.0
            sample.memory_used_ratio = float(psutil.virtual_memory().percent) / 100.0
        except Exception:  # pragma: no cover
            pass
        try:
            import torch

            if torch.cuda.is_available():  # pragma: no cover - hardware dependent
                free, total = torch.cuda.mem_get_info()
                sample.gpu_memory_used = int(total - free)
                sample.gpu_memory_total = int(total)
                if total:
                    sample.gpu_used_ratio = (total - free) / total
        except Exception:
            pass
        with self._lock:
            self.samples.append(sample)
            if len(self.samples) > self.keep:
                del self.samples[: len(self.samples) - self.keep]
        return sample

    def latest(self) -> HardwareSample | None:
        with self._lock:
            return self.samples[-1] if self.samples else None
