"""Backend selection.

The selection is capability based:

1. enumerate the available backends,
2. initialise the candidate (which validates the device and the provider),
3. run a small execution test,
4. select the first backend that passes and report why.

A backend is never chosen solely because its name appears in a device
enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils.errors import BackendError
from .arm_neon.backend import ArmNeonBackend
from .cpu.backend import CpuBackend
from .cuda.backend import CudaBackend
from .intel_gpu.backend import IntelGpuBackend
from .interface import Backend, BackendCapabilities

__all__ = ["BackendSelection", "BackendRegistry", "select_backend", "default_registry"]


@dataclass
class BackendSelection:
    """The outcome of the backend selection."""

    backend: Backend | None = None
    capabilities: BackendCapabilities | None = None
    reason: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.capabilities.name if self.capabilities else "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "reason": self.reason,
            "attempts": list(self.attempts),
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
        }


class BackendRegistry:
    """Creates and selects backends."""

    #: The priority order mandated by the platform design.
    PRIORITY = ("cuda", "intel_gpu", "arm_neon", "cpu")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.selection: BackendSelection | None = None

    # ------------------------------------------------------------------
    def candidates(self) -> list[Backend]:
        """Return the candidate backends in priority order."""

        preferred = str(self.config.get("preferred", "auto")).lower()
        openvino_model = self.config.get("openvino_model")
        created: dict[str, Backend] = {
            "cuda": CudaBackend(),
            "intel_gpu": IntelGpuBackend(model_path=openvino_model),
            "arm_neon": ArmNeonBackend(),
            "cpu": CpuBackend(threads=self.config.get("threads")),
        }
        if preferred in created:
            ordered = [preferred] + [name for name in self.PRIORITY if name != preferred]
        else:
            ordered = list(self.PRIORITY)
        return [created[name] for name in ordered]

    def select(self, run_test: bool = True) -> BackendSelection:
        """Select the first backend that can be initialised and used."""

        selection = BackendSelection()
        for backend in self.candidates():
            try:
                backend.initialize()
            except Exception as exc:
                selection.attempts.append(
                    {"backend": backend.name, "ok": False, "reason": str(exc)}
                )
                continue
            try:
                if run_test:
                    self._execution_test(backend)
            except Exception as exc:
                selection.attempts.append(
                    {"backend": backend.name, "ok": False, "reason": f"execution test failed: {exc}"}
                )
                backend.shutdown()
                continue
            capabilities = backend.capabilities()
            selection.backend = backend
            selection.capabilities = capabilities
            selection.reason = f"selected {capabilities.name}: {capabilities.notes}"
            selection.attempts.append({"backend": backend.name, "ok": True, "reason": selection.reason})
            self.selection = selection
            return selection

        selection.reason = "no backend could be initialised"
        self.selection = selection
        return selection

    def _execution_test(self, backend: Backend) -> None:
        """Run a tiny allocation/execution test on the backend.

        The test does not run the model: the backend is selected before a
        model is placed on it.  It verifies that the backend can allocate
        and operate on a buffer.
        """

        tensor = backend.allocate_tensor((2, 2))
        if tensor is None:
            raise BackendError("the backend could not allocate a tensor")
        try:
            import numpy as np

            values = np.asarray(tensor)
            if values.size:
                values.flat[0] = 1.0
            if float(np.sum(values)) < 0:  # pragma: no cover - defensive
                raise BackendError("the backend produced an invalid buffer")
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"the backend failed the execution test: {exc}") from exc


def default_registry(config: dict[str, Any] | None = None) -> BackendRegistry:
    """Return a registry with the default configuration."""

    return BackendRegistry(config)


def select_backend(config: dict[str, Any] | None = None) -> BackendSelection:
    """Select a backend and return the selection result."""

    return default_registry(config).select()
