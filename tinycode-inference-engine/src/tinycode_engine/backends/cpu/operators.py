"""CPU operator implementations.

The CPU backend uses numpy by default and transparently switches to the
native C++ kernels when the shared library has been built.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...runtime.kernel_dispatch import KernelDispatcher
from ...runtime.operators import OperatorSet

__all__ = ["CpuOperators"]


class CpuOperators(OperatorSet):
    """Operators that use the native kernels when they are available."""

    name = "cpu"

    def __init__(self, dispatcher: KernelDispatcher | None = None, threads: int | None = None) -> None:
        self.dispatcher = dispatcher or KernelDispatcher()
        self.threads = threads

    @property
    def native_available(self) -> bool:
        return self.dispatcher.native_available

    def linear(self, x: Any, weight: Any, bias: Any = None) -> Any:
        if self.dispatcher.native_available and x.ndim == 2 and weight.ndim == 2:
            result = self.dispatcher.matmul(x, weight.T)
            if bias is not None:
                result = result + bias
            return result
        result = np.matmul(x, weight.T)
        if bias is not None:
            result = result + bias
        return result

    def rms_norm(self, x: Any, gain: Any, eps: float = 1e-5) -> Any:
        if self.dispatcher.native_available and x.ndim == 1:
            return self.dispatcher.rms_norm(x, gain, eps=eps)
        variance = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
        return (x * (1.0 / np.sqrt(variance + eps))) * gain

    def silu(self, x: Any) -> Any:
        if self.dispatcher.native_available and x.ndim == 1:
            return self.dispatcher.silu(x)
        values = np.asarray(x, dtype=np.float32)
        return values / (1.0 + np.exp(-values))
