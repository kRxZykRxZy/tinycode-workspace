"""The portable CPU backend.

CPU-only execution is a first class supported mode: it works on every
supported operating system and does not require a compiler.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...model.model_adapter import TinyCodeModel
from ...runtime.kernel_dispatch import KernelDispatcher
from ...runtime.operators import OperatorSet
from ...utils.errors import BackendError
from ..interface import Backend, BackendCapabilities, BackendUnavailableError
from .capabilities import cpu_capabilities
from .operators import CpuOperators
from .threading import configure_threads

__all__ = ["CpuBackend"]


class CpuBackend(Backend):
    """Executes the model with numpy (and the native kernels when built)."""

    name = "cpu"

    def __init__(self, threads: int | None = None, prefer_native: bool = True) -> None:
        super().__init__()
        self.threads = threads
        self.prefer_native = prefer_native
        self.dispatcher = KernelDispatcher(prefer_native=prefer_native)
        self.operators: OperatorSet = CpuOperators(self.dispatcher, threads=threads)
        self._model: TinyCodeModel | None = None
        self._thread_config = configure_threads(threads)

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Prepare the CPU backend."""

        try:
            np.zeros(1, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - numpy always works
            raise BackendUnavailableError(f"The CPU backend is not usable: {exc}") from exc
        self._initialized = True

    def capabilities(self) -> BackendCapabilities:
        capabilities = cpu_capabilities()
        if self.dispatcher.native_available:
            capabilities.notes += f"; native kernels active (v{self.dispatcher.library.version})"
        else:
            capabilities.notes += "; native kernels unavailable, using numpy"
        return capabilities

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
        """Run the model and return the logits of the last position."""

        model = self._model
        if model is None:
            raise BackendError("No model has been loaded into the CPU backend.")
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
            "native_kernels": self.dispatcher.native_available,
        }

    def release_resources(self) -> None:
        self._model = None
        super().release_resources()
