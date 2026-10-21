"""Aggregate hardware capability detection."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from .cuda_info import CudaInfo, detect_cuda
from .memory_info import MemoryInfo, detect_memory

__all__ = ["HardwareProfile", "detect_hardware"]


@dataclass
class HardwareProfile:
    """Everything the builder knows about the machine it runs on."""

    platform: str = ""
    python_version: str = ""
    machine: str = ""
    processor: str = ""
    cpu_count: int = 1
    cuda: CudaInfo = field(default_factory=CudaInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    is_colab: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_gpu(self) -> bool:
        return self.cuda.available

    @property
    def recommended_precision(self) -> str:
        if not self.cuda.available:
            return "float32"
        return "bfloat16" if self.cuda.supports_bf16 else "float16"

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "machine": self.machine,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "cuda": self.cuda.to_dict(),
            "memory": self.memory.to_dict(),
            "is_colab": self.is_colab,
            "recommended_precision": self.recommended_precision,
            "extra": self.extra,
        }


def detect_hardware() -> HardwareProfile:
    """Probe the current machine for GPUs, memory and platform information."""

    profile = HardwareProfile(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        machine=platform.machine(),
        processor=platform.processor(),
        cuda=detect_cuda(),
        memory=detect_memory(),
    )
    try:
        import os

        profile.cpu_count = os.cpu_count() or profile.cpu_count
        profile.is_colab = "google.colab" in sys.modules or Path_colab_marker_exists()
    except Exception:  # pragma: no cover
        profile.is_colab = False
    return profile


def Path_colab_marker_exists() -> bool:
    """Return whether the environment looks like a Google Colab runtime."""

    import os

    return os.path.exists("/content/drive") or os.path.exists("/content/colab")
