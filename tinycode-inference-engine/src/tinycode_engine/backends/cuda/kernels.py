"""Which CUDA kernels have a dedicated implementation.

Not every operator needs a custom CUDA kernel: PyTorch already provides
optimised implementations for the operations that dominate the runtime.
This module records which path each operator takes so that the backend
can report it honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["KernelPlan", "cuda_kernel_plan"]


@dataclass
class KernelPlan:
    """The execution path of every operator."""

    paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.paths)


def cuda_kernel_plan() -> KernelPlan:
    """Return the kernel plan of the CUDA backend."""

    return KernelPlan(
        paths={
            "embedding": "torch.nn.functional.embedding",
            "linear": "torch.matmul (cuBLAS)",
            "layernorm": "torch.nn.functional.layer_norm",
            "rmsnorm": "custom fused torch implementation",
            "causal_attention": "torch.nn.functional.scaled_dot_product_attention",
            "softmax": "torch.softmax",
            "silu": "torch.nn.functional.silu",
            "sampling": "torch multinomial / argmax",
        }
    )
