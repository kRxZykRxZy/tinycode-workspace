"""The CUDA backend.

The backend is only usable when PyTorch is installed *and* a CUDA device
is present.  It reports an out-of-memory condition explicitly and can fall
back to the CPU backend at the registry level.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...utils.errors import BackendError
from ..interface import Backend, BackendCapabilities, BackendUnavailableError
from .capabilities import detect_cuda_capabilities
from .kernels import cuda_kernel_plan
from .memory import gpu_memory_snapshot
from .operators import TorchOperators, TorchTinyCodeModel

__all__ = ["CudaBackend"]


class CudaBackend(Backend):
    """Executes the model on an NVIDIA GPU through PyTorch."""

    name = "cuda"

    def __init__(self, device_index: int = 0, precision: str = "float16") -> None:
        super().__init__()
        self.device_index = device_index
        self.requested_precision = precision
        self.capabilities_info = detect_cuda_capabilities()
        self.operators = TorchOperators()
        self._model: TorchTinyCodeModel | None = None
        self._device: Any = None
        self._precision = precision

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Initialise the CUDA context."""

        if not self.capabilities_info.available:
            raise BackendUnavailableError(
                f"The CUDA backend is unavailable: {self.capabilities_info.reason}"
            )
        try:
            import torch

            self._device = torch.device(f"cuda:{self.device_index}")
            torch.cuda.set_device(self._device)
            # A tiny allocation proves that the device is usable.
            torch.zeros(1, device=self._device)
        except Exception as exc:
            raise BackendUnavailableError(f"CUDA initialisation failed: {exc}") from exc
        self._initialized = True

    def capabilities(self) -> BackendCapabilities:
        precisions = self.capabilities_info.supported_precisions
        return BackendCapabilities(
            name="cuda",
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
            supported_precisions=precisions,
            supports_batching=False,
            supports_kv_cache=True,
            device_memory_bytes=self.capabilities_info.total_memory_bytes,
            is_accelerated=True,
            execution_provider="pytorch-cuda",
            notes="; ".join(
                [
                    f"devices: {self.capabilities_info.device_count}",
                    f"runtime: {self.capabilities_info.cuda_runtime}",
                    f"kernels: {cuda_kernel_plan().to_dict()}",
                ]
            ),
        )

    def load_model(self, weights: dict[str, Any], config: dict[str, Any]) -> None:
        super().load_model(weights, config)
        if self._device is None:
            raise BackendError("The CUDA backend has not been initialised.")
        self._model = TorchTinyCodeModel(config, weights, device=self._device)

    def allocate_tensor(self, shape: tuple[int, ...], dtype: Any = "float32") -> Any:
        import torch

        return torch.zeros(shape, dtype=dtype, device=self._device)

    def execute_forward(
        self,
        input_ids: Any,
        weights: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        cache: Any = None,
    ) -> Any:
        """Run the model on the GPU and return numpy logits."""

        if self._model is None:
            raise BackendError("No model has been loaded into the CUDA backend.")
        import torch

        try:
            with torch.no_grad():
                logits = self._model.forward(input_ids, cache=cache)
            return logits.detach().float().cpu().numpy()
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
            raise BackendError(f"CUDA out of memory: {exc}") from exc

    def synchronize(self) -> None:
        try:
            import torch

            if self._device is not None:
                torch.cuda.synchronize(self._device)
        except Exception:  # pragma: no cover
            pass

    def report_memory(self) -> dict[str, Any]:
        snapshot = gpu_memory_snapshot(self.device_index)
        snapshot["backend"] = self.name
        return snapshot

    def release_resources(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
        super().release_resources()
