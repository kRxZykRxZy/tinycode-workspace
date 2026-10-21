"""Detection of NVIDIA CUDA devices (best effort, never fatal)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CudaInfo", "detect_cuda"]


@dataclass
class CudaInfo:
    """What the builder could learn about CUDA on this machine."""

    available: bool = False
    device_count: int = 0
    devices: list[dict[str, Any]] = field(default_factory=list)
    torch_version: str | None = None
    cuda_runtime: str | None = None
    cudnn_version: str | None = None
    reason: str = "not probed"

    @property
    def supports_bf16(self) -> bool:
        return any(bool(dev.get("supports_bf16")) for dev in self.devices)

    @property
    def supports_fp16(self) -> bool:
        return any(bool(dev.get("supports_fp16")) for dev in self.devices)

    @property
    def total_vram_bytes(self) -> int:
        return int(sum(int(dev.get("total_memory", 0)) for dev in self.devices))

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "device_count": self.device_count,
            "devices": self.devices,
            "torch_version": self.torch_version,
            "cuda_runtime": self.cuda_runtime,
            "cudnn_version": self.cudnn_version,
            "reason": self.reason,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
        }


def detect_cuda() -> CudaInfo:
    """Probe for CUDA devices.

    This never raises: an absent or broken CUDA installation is reported as
    "not available" together with a reason.
    """

    info = CudaInfo()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        info.reason = f"PyTorch import failed: {exc}"
        return info

    info.torch_version = getattr(torch, "__version__", None)
    try:
        info.cuda_runtime = getattr(torch.version, "cuda", None)
        info.cudnn_version = (
            torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        )
    except Exception:  # pragma: no cover
        pass

    try:
        if not torch.cuda.is_available():
            info.reason = "torch.cuda.is_available() returned False"
            return info
        info.available = True
        info.device_count = torch.cuda.device_count()
        for index in range(info.device_count):
            props = torch.cuda.get_device_properties(index)
            info.devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": int(props.total_memory),
                    "capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
                    "supports_fp16": bool(getattr(props, "major", 0) >= 6),
                    "supports_bf16": bool(getattr(props, "major", 0) >= 8),
                }
            )
        info.reason = "cuda devices detected"
    except Exception as exc:  # pragma: no cover - hardware dependent
        info.available = False
        info.reason = f"cuda probe failed: {exc}"
    return info
