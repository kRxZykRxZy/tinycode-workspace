"""Static memory estimation for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MemoryEstimate", "estimate_training_memory", "dtype_bytes"]


def dtype_bytes(dtype: str) -> int:
    """Return the number of bytes per element for a precision name."""

    return {
        "float32": 4,
        "fp32": 4,
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
        "int8": 1,
    }.get((dtype or "float32").lower(), 4)


@dataclass
class MemoryEstimate:
    """Estimated memory consumption of a training step."""

    parameter_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0
    activation_bytes: int = 0
    total_bytes: int = 0
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_bytes": self.parameter_bytes,
            "gradient_bytes": self.gradient_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "activation_bytes": self.activation_bytes,
            "total_bytes": self.total_bytes,
            "total_gb": round(self.total_gb, 4),
            "notes": list(self.notes),
        }


def estimate_training_memory(
    parameter_count: int,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    n_layers: int,
    precision: str = "float32",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> MemoryEstimate:
    """Estimate the memory needed for one training step."""

    element = dtype_bytes(precision)
    parameter_bytes = parameter_count * element
    gradient_bytes = parameter_count * element

    if optimizer.lower() == "adamw":
        # two moments stored in fp32
        optimizer_bytes = parameter_count * 4 * 2
    elif optimizer.lower() == "adam":
        optimizer_bytes = parameter_count * 4 * 2
    else:
        optimizer_bytes = 0

    activation_factor = 1 if gradient_checkpointing else 2
    activation_bytes = (
        batch_size
        * sequence_length
        * d_model
        * n_layers
        * activation_factor
        * element
    )

    notes: list[str] = []
    if gradient_checkpointing:
        notes.append("Gradient checkpointing enabled: activations are recomputed.")
    if precision in {"float16", "fp16"}:
        notes.append("float16 training additionally needs a gradient scaler (small overhead).")

    total = parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes
    return MemoryEstimate(
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        total_bytes=total,
        notes=notes,
    )
