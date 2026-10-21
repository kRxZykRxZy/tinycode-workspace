"""Model architecture configuration.

The architecture is fully described by a small, versioned dictionary that is
embedded in checkpoints and in the exported artifact manifest.  The
inference engine re-implements the same forward pass from this
description, therefore the field names below are part of the cross
repository contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..configuration.immutable_constraints import CONTEXT_LENGTH, assert_context_length

__all__ = [
    "ARCHITECTURE_ID",
    "NORM_TYPES",
    "ACTIVATIONS",
    "POSITION_ENCODINGS",
    "TransformerConfig",
]

#: Identifier of the architecture family understood by the inference engine.
ARCHITECTURE_ID = "tinycode-decoder-v1"

NORM_TYPES = ("layernorm", "rmsnorm")
ACTIVATIONS = ("silu", "gelu")
POSITION_ENCODINGS = ("learned", "sinusoidal")


@dataclass
class TransformerConfig:
    """Configuration of the TinyCode causal decoder."""

    vocab_size: int = 98_304
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    context_length: int = CONTEXT_LENGTH
    norm_type: str = "layernorm"
    norm_eps: float = 1e-5
    activation: str = "silu"
    position_encoding: str = "learned"
    qkv_bias: bool = True
    ff_bias: bool = True
    head_bias: bool = False
    tie_embeddings: bool = False
    architecture_id: str = ARCHITECTURE_ID
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self) -> "TransformerConfig":
        """Validate the configuration, raising ``ValueError`` on problems."""

        assert_context_length(self.context_length)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0 or self.d_ff <= 0:
            raise ValueError("d_model, n_layers, n_heads and d_ff must all be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.norm_type not in NORM_TYPES:
            raise ValueError(f"norm_type must be one of {NORM_TYPES}")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.position_encoding not in POSITION_ENCODINGS:
            raise ValueError(f"position_encoding must be one of {POSITION_ENCODINGS}")
        return self

    # ------------------------------------------------------------------
    # derived properties
    # ------------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ff_ratio(self) -> float:
        return self.d_ff / self.d_model if self.d_model else 0.0

    @property
    def norm_parameters_per_norm(self) -> int:
        return 2 * self.d_model if self.norm_type == "layernorm" else self.d_model

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the contract-facing description of this configuration."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformerConfig":
        """Rebuild a configuration from its dictionary form."""

        allowed = {
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "d_ff",
            "context_length",
            "norm_type",
            "norm_eps",
            "activation",
            "position_encoding",
            "qkv_bias",
            "ff_bias",
            "head_bias",
            "tie_embeddings",
            "architecture_id",
            "notes",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown architecture fields: {sorted(unknown)}")
        config = cls(**{key: payload[key] for key in allowed if key in payload})
        return config

    def fingerprint(self) -> dict[str, Any]:
        """Return the subset of fields that define the computation graph."""

        payload = self.to_dict()
        payload.pop("notes", None)
        return payload
