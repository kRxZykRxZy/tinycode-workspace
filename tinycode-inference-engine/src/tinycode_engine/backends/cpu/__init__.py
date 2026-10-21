"""The portable CPU backend."""

from __future__ import annotations

from .backend import CpuBackend
from .capabilities import cpu_capabilities
from .operators import CpuOperators
from .threading import ThreadConfig, configure_threads, current_thread_settings
from .vectorization import VectorizationInfo, detect_vectorization

__all__ = [
    "CpuBackend",
    "CpuOperators",
    "ThreadConfig",
    "VectorizationInfo",
    "configure_threads",
    "cpu_capabilities",
    "current_thread_settings",
    "detect_vectorization",
]
