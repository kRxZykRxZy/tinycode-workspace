"""The CUDA backend."""

from __future__ import annotations

from .backend import CudaBackend
from .capabilities import CudaCapabilities, detect_cuda_capabilities
from .kernels import KernelPlan, cuda_kernel_plan
from .memory import free_memory_bytes, gpu_memory_snapshot, total_memory_bytes
from .operators import TorchOperators, TorchTinyCodeModel

__all__ = [
    "CudaBackend",
    "CudaCapabilities",
    "KernelPlan",
    "TorchOperators",
    "TorchTinyCodeModel",
    "cuda_kernel_plan",
    "detect_cuda_capabilities",
    "free_memory_bytes",
    "gpu_memory_snapshot",
    "total_memory_bytes",
]
