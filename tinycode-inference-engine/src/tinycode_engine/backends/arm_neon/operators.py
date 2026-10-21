"""NEON accelerated operators.

The native library is compiled with the NEON code path on ARM hosts.  When
it is not available the operators fall back to the numpy implementation;
no ARM specific instruction is ever executed on another architecture.
"""

from __future__ import annotations

from typing import Any

from ...runtime.kernel_dispatch import KernelDispatcher
from ...runtime.operators import OperatorSet
from .capabilities import NeonCapabilities, detect_neon

__all__ = ["NeonOperators"]


class NeonOperators(OperatorSet):
    """Operators that use the native NEON kernels when available."""

    name = "arm_neon"

    def __init__(self, dispatcher: KernelDispatcher | None = None) -> None:
        self.dispatcher = dispatcher or KernelDispatcher()
        self.neon: NeonCapabilities = detect_neon()

    @property
    def native_available(self) -> bool:
        return self.dispatcher.native_available and self.neon.available

    def linear(self, x: Any, weight: Any, bias: Any = None) -> Any:
        if self.native_available and x.ndim == 2 and weight.ndim == 2:
            result = self.dispatcher.matmul(x, weight.T)
            if bias is not None:
                result = result + bias
            return result
        result = x @ weight.T
        if bias is not None:
            result = result + bias
        return result

    def rms_norm(self, x: Any, gain: Any, eps: float = 1e-5) -> Any:
        if self.native_available and x.ndim == 1:
            return self.dispatcher.rms_norm(x, gain, eps=eps)
        variance = (x * x).mean(axis=-1, keepdims=True)
        return x / (variance + eps) ** 0.5 * gain

    def silu(self, x: Any) -> Any:
        import numpy as np

        if self.native_available and x.ndim == 1:
            return self.dispatcher.silu(x)
        values = np.asarray(x, dtype=np.float32)
        return values / (1.0 + np.exp(-values))
