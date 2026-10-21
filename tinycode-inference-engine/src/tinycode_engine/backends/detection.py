"""Device discovery.

Discovery only *enumerates* what is present.  Whether a device can be
used is decided by the registry, which initialises the backend and runs a
small execution test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils.platform import detect_architecture, detect_simd
from .arm_neon.capabilities import detect_neon
from .cuda.capabilities import detect_cuda_capabilities
from .intel_gpu.provider import detect_graphics_devices, detect_openvino

__all__ = ["DeviceInventory", "discover_devices"]


@dataclass
class DeviceInventory:
    """Everything that was found on this machine."""

    cuda: dict[str, Any] = field(default_factory=dict)
    graphics: list[dict[str, Any]] = field(default_factory=list)
    openvino: dict[str, Any] = field(default_factory=dict)
    neon: dict[str, Any] = field(default_factory=dict)
    architecture: str = ""
    simd: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda": self.cuda,
            "graphics": list(self.graphics),
            "openvino": self.openvino,
            "neon": self.neon,
            "architecture": self.architecture,
            "simd": list(self.simd),
        }


def discover_devices() -> DeviceInventory:
    """Enumerate the devices of the machine."""

    return DeviceInventory(
        cuda=detect_cuda_capabilities().to_dict(),
        graphics=[device.to_dict() for device in detect_graphics_devices()],
        openvino=detect_openvino().to_dict(),
        neon=detect_neon().to_dict(),
        architecture=detect_architecture(),
        simd=detect_simd(),
    )
