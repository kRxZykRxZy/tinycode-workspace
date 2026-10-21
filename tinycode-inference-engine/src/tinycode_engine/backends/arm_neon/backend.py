"""The ARM NEON backend.

The backend is only selected on ARM CPUs that actually support NEON.  It
uses the native kernels when they have been built; otherwise it falls back
to the portable CPU implementation while still reporting the NEON
capability of the machine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...model.model_adapter import TinyCodeModel
from ...runtime.kernel_dispatch import KernelDispatcher
from ...runtime.operators import OperatorSet
from ...utils.errors import BackendError
from ..interface import Backend, BackendCapabilities, BackendUnavailableError
from .capabilities import NeonCapabilities, detect_neon
from .operators import NeonOperators

__all__ = ["ArmNeonBackend"]


class ArmNeonBackend(Backend):
    """Executes the model on an ARM CPU with NEON acceleration."""

    name = "arm_neon"

    def __init__(self, prefer_native: bool = True) -> None:
        super().__init__()
        self.neon: NeonCapabilities = detect_neon()
        self.dispatcher = KernelDispatcher(prefer_native=prefer_native)
        self.operators: OperatorSet = NeonOperators(self.dispatcher)
        self._model: TinyCodeModel | None = None

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Prepare the NEON backend."""

        if not self.neon.available:
            raise BackendUnavailableError(
                f"The ARM NEON backend is unavailable: {self.neon.reason}"
            )
        self._initialized = True

    def capabilities(self) -> BackendCapabilities:
        notes = [f"NEON: {self.neon.variant or 'available'}"]
        if self.dispatcher.native_available:
            notes.append("native NEON kernels active")
        else:
            notes.append("native kernels unavailable, using the portable CPU path")
        return BackendCapabilities(
            name="arm_neon",
            supported_operators=frozenset(
                {
                    "embedding",
                    "linear",
                    "layernorm",
                    "rmsnorm",
                    "causal_attention",
                    "softmax",
                    "silu",
                    "gelu",
                    "sampling",
                }
            ),
            supported_precisions=frozenset({"float32"}),
            supports_batching=False,
            supports_kv_cache=True,
            device_memory_bytes=0,
            is_accelerated=self.dispatcher.native_available,
            execution_provider="native-neon",
            notes="; ".join(notes),
        )

    def load_model(self, weights: dict[str, Any], config: dict[str, Any]) -> None:
        super().load_model(weights, config)
        self._model = TinyCodeModel(config, weights, operators=self.operators)

    def allocate_tensor(self, shape: tuple[int, ...], dtype: Any = "float32") -> Any:
        return np.zeros(shape, dtype=dtype)

    def execute_forward(
        self,
        input_ids: Any,
        weights: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        cache: Any = None,
    ) -> Any:
        model = self._model
        if model is None:
            raise BackendError("No model has been loaded into the ARM NEON backend.")
        return model.forward(input_ids, cache=cache)

    def report_memory(self) -> dict[str, Any]:
        weight_bytes = int(
            sum(
                array.nbytes
                for array in self._weights.values()
                if isinstance(array, np.ndarray)
            )
        )
        return {
            "backend": self.name,
            "allocated_bytes": weight_bytes,
            "reserved_bytes": weight_bytes,
            "neon": self.neon.to_dict(),
        }

    def release_resources(self) -> None:
        self._model = None
        super().release_resources()
