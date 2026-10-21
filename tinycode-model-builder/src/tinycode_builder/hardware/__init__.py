"""Hardware detection and capability reporting for the training environment."""

from __future__ import annotations

from .capability_report import CapabilityReport, build_capability_report, format_capability_report
from .cuda_info import CudaInfo, detect_cuda
from .detection import detect_hardware
from .memory_info import MemoryInfo, detect_memory

__all__ = [
    "CapabilityReport",
    "CudaInfo",
    "MemoryInfo",
    "build_capability_report",
    "detect_cuda",
    "detect_hardware",
    "detect_memory",
    "format_capability_report",
]
