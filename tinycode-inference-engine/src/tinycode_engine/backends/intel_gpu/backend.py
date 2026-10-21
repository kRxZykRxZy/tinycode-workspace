"""The Intel integrated graphics backend.

The backend requires

1. a verified OpenVINO execution provider with a *usable* GPU plugin
   (a trivial model is compiled on the ``GPU`` device to prove it),
2. an OpenVINO IR export of the model, and
3. a compilation that really lands on the requested device.

The backend reports the device that was actually used.  When the
compilation falls back to the CPU the backend says so; it never claims
GPU acceleration while the work runs on the CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...utils.errors import BackendError
from ..interface import Backend, BackendCapabilities, BackendUnavailableError
from .capabilities import intel_gpu_status
from .operators import OpenVinoOperators
from .provider import import_openvino

__all__ = ["IntelGpuBackend"]


class IntelGpuBackend(Backend):
    """Executes an OpenVINO IR model on Intel integrated graphics."""

    name = "intel_gpu"

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "GPU",
        allow_cpu_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.allow_cpu_fallback = allow_cpu_fallback
        self.status = intel_gpu_status()
        self._compiled: Any = None
        self._operators: OpenVinoOperators | None = None
        self._used_device: str | None = None
        self._input_name: str = "tokens"
        self._input_dtype: Any = np.int64

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Initialise the OpenVINO runtime and compile the model."""

        provider = self.status.get("provider", {})
        if not provider.get("available"):
            raise BackendUnavailableError(
                f"The Intel GPU backend is unavailable: {self.status['reason']}"
            )

        requested_device = self.device
        if requested_device.upper().startswith("GPU") and not provider.get("gpu_plugin"):
            # The GPU plugin is not available on this machine.  Loading it
            # anyway can abort the process, so the backend either refuses
            # or falls back to the CPU explicitly.
            if not self.allow_cpu_fallback:
                raise BackendUnavailableError(
                    "The Intel GPU backend is unavailable: " f"{self.status['reason']}"
                )
            self.device = "CPU"

        try:
            ov = import_openvino()
        except Exception as exc:  # pragma: no cover - provider dependent
            raise BackendUnavailableError(f"OpenVINO is not usable: {exc}") from exc

        if self.model_path is None or not self.model_path.with_suffix(".xml").is_file():
            raise BackendUnavailableError(
                "No OpenVINO IR model was provided; the Intel GPU backend cannot "
                "execute the TinyCode architecture yet."
            )

        try:
            core = ov.Core()
            model = core.read_model(str(self.model_path.with_suffix(".xml")))
            self._compiled = core.compile_model(model, self.device)
        except Exception as exc:
            raise BackendUnavailableError(f"OpenVINO compilation failed: {exc}") from exc

        self._used_device = self._detect_used_device()
        self._operators = OpenVinoOperators(self._compiled)
        self._input_name = self._detect_input_name()
        self._input_dtype = self._detect_input_dtype()
        self._initialized = True

    # ------------------------------------------------------------------
    def _detect_used_device(self) -> str | None:
        """Return the device the compiled model really runs on."""

        try:
            devices = self._compiled.get_property("EXECUTION_DEVICES")
        except Exception:
            return None
        if isinstance(devices, str):
            return devices
        if isinstance(devices, (list, tuple)) and devices:
            return str(devices[0])
        return None

    def _detect_input_dtype(self) -> Any:
        """Return the element type the compiled graph expects for the tokens."""

        try:
            node = list(self._compiled.inputs)[0]
            element_type = str(node.get_element_type()).lower()
            if "32" in element_type:
                return np.int32
        except Exception:
            pass
        return np.int64

    def _detect_input_name(self) -> str:
        """Return the name of the input the compiled graph expects."""

        try:
            nodes = list(self._compiled.inputs)
        except Exception:
            return "tokens"
        for node in nodes:
            for attribute in ("get_any_name", "get_friendly_name"):
                getter = getattr(node, attribute, None)
                if getter is None:
                    continue
                try:
                    name = getter()
                except Exception:
                    continue
                if name:
                    return str(name)
        return "tokens"

    # ------------------------------------------------------------------
    def capabilities(self) -> BackendCapabilities:
        operators: frozenset[str] = frozenset()
        if self._operators is not None:
            operators = frozenset(
                operator
                for operator in (
                    "embedding",
                    "linear",
                    "layernorm",
                    "rmsnorm",
                    "causal_attention",
                    "softmax",
                    "silu",
                    "sampling",
                )
                if self._operators.supports(operator)
            )

        used = self._used_device or "unknown"
        accelerated = "GPU" in used.upper()
        notes = [
            self.status["reason"],
            f"requested device: {self.device}",
            f"device actually used: {used}",
        ]
        if not accelerated:
            notes.append(
                "the execution did not land on the GPU; the backend does not claim "
                "GPU acceleration"
            )
        return BackendCapabilities(
            name="intel_gpu",
            supported_operators=operators,
            supported_precisions=frozenset({"float32", "float16"}),
            supports_batching=False,
            supports_kv_cache=False,
            device_memory_bytes=0,
            is_accelerated=accelerated,
            execution_provider=f"openvino:{used}",
            notes="; ".join(notes),
        )

    def load_model(self, weights: dict[str, Any], config: dict[str, Any]) -> None:
        super().load_model(weights, config)

    def allocate_tensor(self, shape: tuple[int, ...], dtype: Any = "float32") -> Any:
        return np.zeros(shape, dtype=dtype)

    def execute_forward(
        self,
        input_ids: Any,
        weights: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        cache: Any = None,
    ) -> Any:
        """Execute the compiled OpenVINO graph."""

        if self._compiled is None:
            raise BackendError("The Intel GPU backend has not been initialised.")
        ov = import_openvino()
        tokens = np.asarray(input_ids, dtype=self._input_dtype).reshape(1, -1)
        try:
            request = self._compiled.create_infer_request()
            tensor = ov.Tensor(tokens)
            request.infer({self._input_name: tensor})
            output = request.get_output_tensor(0).data
            return np.asarray(output).reshape(-1)
        except Exception as exc:
            raise BackendError(f"OpenVINO inference failed: {exc}") from exc

    def report_memory(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "device": self._used_device,
        }

    def release_resources(self) -> None:
        self._compiled = None
        self._operators = None
        self._used_device = None
        super().release_resources()
