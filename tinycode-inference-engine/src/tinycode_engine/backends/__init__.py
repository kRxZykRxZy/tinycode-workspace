"""Hardware backends of the inference engine."""

from __future__ import annotations

from .arm_neon.backend import ArmNeonBackend
from .cpu.backend import CpuBackend
from .cuda.backend import CudaBackend
from .detection import DeviceInventory, discover_devices
from .intel_gpu.backend import IntelGpuBackend
from .interface import Backend, BackendCapabilities, BackendUnavailableError
from .registry import BackendRegistry, BackendSelection, default_registry, select_backend

__all__ = [
    "ArmNeonBackend",
    "Backend",
    "BackendCapabilities",
    "BackendRegistry",
    "BackendSelection",
    "BackendUnavailableError",
    "CpuBackend",
    "CudaBackend",
    "DeviceInventory",
    "IntelGpuBackend",
    "default_registry",
    "discover_devices",
    "select_backend",
]
