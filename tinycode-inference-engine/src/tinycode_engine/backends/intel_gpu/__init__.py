"""The Intel integrated graphics backend."""

from __future__ import annotations

from .backend import IntelGpuBackend
from .capabilities import intel_gpu_capabilities, intel_gpu_status
from .operators import OpenVinoOperators
from .provider import (
    GraphicsDevice,
    ProviderStatus,
    detect_graphics_devices,
    detect_intel_graphics,
    detect_openvino,
)

__all__ = [
    "GraphicsDevice",
    "IntelGpuBackend",
    "OpenVinoOperators",
    "ProviderStatus",
    "detect_graphics_devices",
    "detect_intel_graphics",
    "detect_openvino",
    "intel_gpu_capabilities",
    "intel_gpu_status",
]
