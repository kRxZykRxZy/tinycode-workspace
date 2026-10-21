"""CUDA capability detection.

The detection never assumes that a CUDA device exists: PyTorch must be
installed *and* report a usable device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CudaCapabilities", "detect_cuda_capabilities"]


@dataclass
class CudaCapabilities:
    """What the CUDA backend can do on this machine."""

    available: bool = False
    device_count: int = 0
    devices: list[dict[str, Any]] = field(default_factory=list)
    torch_version: str | None = None
    cuda_runtime: str | None = None
    reason: str = ""

    @property
    def supported_precisions(self) -> frozenset[str]:
        if not self.available:
            return frozenset()
        precisions = {"float32"}
        for device in self.devices:
            if device.get("supports_fp16"):
                precisions.add("float16")
            if device.get("supports_bf16"):
                precisions.add("bfloat16")
        return frozenset(precisions)

    @property
    def total_memory_bytes(self) -> int:
        return int(sum(int(device.get("total_memory", 0)) for device in self.devices))

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "device_count": self.device_count,
            "devices": list(self.devices),
            "torch_version": self.torch_version,
            "cuda_runtime": self.cuda_runtime,
            "reason": self.reason,
            "supported_precisions": sorted(self.supported_precisions),
        }


def detect_cuda_capabilities() -> CudaCapabilities:
    """Probe for CUDA devices."""

    capabilities = CudaCapabilities()
    try:
        import torch
    except Exception as exc:
        capabilities.reason = f"PyTorch is not installed: {exc}"
        return capabilities

    capabilities.torch_version = getattr(torch, "__version__", None)
    capabilities.cuda_runtime = getattr(torch.version, "cuda", None)

    try:
        if not torch.cuda.is_available():
            capabilities.reason = "torch.cuda.is_available() returned False"
            return capabilities
        capabilities.available = True
        capabilities.device_count = torch.cuda.device_count()
        for index in range(capabilities.device_count):
            properties = torch.cuda.get_device_properties(index)
            capabilities.devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory": int(properties.total_memory),
                    "capability": f"{properties.major}.{properties.minor}",
                    "supports_fp16": bool(properties.major >= 6),
                    "supports_bf16": bool(properties.major >= 8),
                }
            )
        capabilities.reason = "CUDA devices detected"
    except Exception as exc:  # pragma: no cover - hardware dependent
        capabilities.available = False
        capabilities.reason = f"CUDA probe failed: {exc}"
    return capabilities
