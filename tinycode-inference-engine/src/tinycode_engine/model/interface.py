"""The interface a served model must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["ModelInterface", "ModelCapabilities"]


class ModelCapabilities:
    """Capabilities of a loaded model."""

    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        parameter_count: int = 0,
        supports_kv_cache: bool = True,
        supports_batching: bool = True,
        name: str = "tinycode",
    ) -> None:
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.parameter_count = parameter_count
        self.supports_kv_cache = supports_kv_cache
        self.supports_batching = supports_batching
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "parameter_count": self.parameter_count,
            "supports_kv_cache": self.supports_kv_cache,
            "supports_batching": self.supports_batching,
        }


class ModelInterface(ABC):
    """A model that the runtime can execute."""

    name: str = "tinycode"

    @abstractmethod
    def forward(self, input_ids: Any, cache: Any = None) -> Any:
        """Run the model and return the logits of the last position."""

    @abstractmethod
    def reset_cache(self) -> None:
        """Reset the internal state (used before a new request)."""

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return the capabilities of the model."""

    @property
    def context_length(self) -> int:
        return self.capabilities().context_length
