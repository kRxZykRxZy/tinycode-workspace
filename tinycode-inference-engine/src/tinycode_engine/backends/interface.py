"""The hardware backend interface.

A backend owns the actual execution of the model.  It advertises exactly
the operators, precisions and features it really supports; the registry
never selects a backend purely because a device name appears in an
enumeration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BackendCapabilities",
    "Backend",
    "OPERATORS",
    "PRECISIONS",
    "BackendUnavailableError",
]

#: The operators a backend may implement.
OPERATORS: frozenset[str] = frozenset(
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
        "rope",
    }
)

#: The precisions a backend may support.
PRECISIONS: frozenset[str] = frozenset({"float32", "float16", "bfloat16", "int8", "int4"})


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot be used on this machine."""


@dataclass
class BackendCapabilities:
    """What a backend can actually do."""

    name: str = "unknown"
    supported_operators: frozenset[str] = field(default_factory=frozenset)
    supported_precisions: frozenset[str] = field(default_factory=lambda: frozenset({"float32"}))
    supports_batching: bool = False
    supports_kv_cache: bool = True
    device_memory_bytes: int = 0
    is_accelerated: bool = False
    execution_provider: str = ""
    notes: str = ""

    def supports(self, operator: str) -> bool:
        return operator in self.supported_operators

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "supported_operators": sorted(self.supported_operators),
            "supported_precisions": sorted(self.supported_precisions),
            "supports_batching": self.supports_batching,
            "supports_kv_cache": self.supports_kv_cache,
            "device_memory_bytes": self.device_memory_bytes,
            "is_accelerated": self.is_accelerated,
            "execution_provider": self.execution_provider,
            "notes": self.notes,
        }


class Backend(ABC):
    """Interface implemented by every execution backend."""

    name: str = "backend"

    def __init__(self) -> None:
        self._initialized = False
        self._weights: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def initialize(self) -> None:
        """Prepare the backend for execution.

        Raises:
            BackendUnavailableError: when the backend cannot be used.
        """

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of the backend."""

    def load_model(self, weights: dict[str, Any], config: dict[str, Any]) -> None:
        """Load the model weights into the backend."""

        self._weights = weights
        self._config = config

    def release_resources(self) -> None:
        """Release the loaded weights and any device memory."""

        self._weights = {}

    def shutdown(self) -> None:
        """Shut the backend down."""

        self.release_resources()
        self._initialized = False

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    @abstractmethod
    def allocate_tensor(self, shape: tuple[int, ...], dtype: Any = "float32") -> Any:
        """Allocate a tensor of the given shape and dtype."""

    @abstractmethod
    def execute_forward(
        self,
        input_ids: Any,
        weights: dict[str, Any],
        config: dict[str, Any],
        cache: Any = None,
    ) -> Any:
        """Run one forward pass and return the logits of the last position."""

    def execute_attention(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the attention operator (backend specific)."""

        raise BackendUnavailableError(f"{self.name} does not implement a dedicated attention path.")

    def execute_projection(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the projection operator (backend specific)."""

        raise BackendUnavailableError(f"{self.name} does not implement a dedicated projection path.")

    def synchronize(self) -> None:
        """Wait for queued work to finish."""

    def report_memory(self) -> dict[str, Any]:
        """Return the memory usage of the backend."""

        return {"backend": self.name, "allocated_bytes": 0, "reserved_bytes": 0}

    # ------------------------------------------------------------------
    @property
    def initialized(self) -> bool:
        return self._initialized

    def __enter__(self) -> "Backend":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.shutdown()
        return False
