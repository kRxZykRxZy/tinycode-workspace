"""Safeguards: the limits a user request can never exceed.

The safeguards are applied *after* the user request was parsed so that a
requested budget can never bypass the server side limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.errors import RequestError
from .controller import BudgetLimits

__all__ = ["SafeguardPolicy", "Safeguards", "MAX_REQUEST_BYTES"]

#: The maximum size of a request body in bytes.
MAX_REQUEST_BYTES = 1 << 20  # 1 MiB


@dataclass
class SafeguardPolicy:
    """The server side limits."""

    limits: BudgetLimits = None  # type: ignore[assignment]
    max_request_bytes: int = MAX_REQUEST_BYTES
    max_tools: int = 32
    max_stop_sequences: int = 16
    max_stop_sequence_length: int = 256

    def __post_init__(self) -> None:
        if self.limits is None:
            self.limits = BudgetLimits()


class Safeguards:
    """Validates and clamps the user supplied limits."""

    def __init__(self, policy: SafeguardPolicy | None = None) -> None:
        self.policy = policy or SafeguardPolicy()

    # ------------------------------------------------------------------
    def clamp_max_tokens(self, requested: int | None, default: int = 256) -> int:
        """Clamp the requested output budget."""

        if requested is None:
            requested = default
        if requested <= 0:
            raise RequestError("max_tokens must be greater than zero.")
        return int(min(requested, self.policy.limits.max_generation_tokens))

    def clamp_compute_budget(self, requested: int | None) -> int | None:
        """Clamp a user supplied computation budget."""

        if requested is None:
            return None
        if requested <= 0:
            raise RequestError("The computation budget must be greater than zero.")
        return int(min(requested, self.policy.limits.max_computation_tokens))

    def clamp_latency_ms(self, requested: float | None) -> float | None:
        """Clamp a user supplied latency target."""

        if requested is None:
            return None
        if requested <= 0:
            raise RequestError("The latency target must be greater than zero.")
        return float(min(requested, self.policy.limits.max_latency_ms))

    def validate_stop_sequences(self, sequences: list[str] | None) -> tuple[str, ...]:
        """Validate the requested stop sequences."""

        if not sequences:
            return ()
        if len(sequences) > self.policy.max_stop_sequences:
            raise RequestError(
                f"At most {self.policy.max_stop_sequences} stop sequences are supported."
            )
        for sequence in sequences:
            if len(sequence) > self.policy.max_stop_sequence_length:
                raise RequestError(
                    "A stop sequence is longer than "
                    f"{self.policy.max_stop_sequence_length} characters."
                )
        return tuple(sequences)

    def validate_body_size(self, size: int) -> None:
        """Reject requests that are too large."""

        if size > self.policy.max_request_bytes:
            raise RequestError(
                f"The request body is larger than the limit of "
                f"{self.policy.max_request_bytes} bytes."
            )

    def validate_tools(self, tools: list[Any] | None) -> list[Any]:
        """Reject unsupported tool definitions."""

        if not tools:
            return []
        if len(tools) > self.policy.max_tools:
            raise RequestError(f"At most {self.policy.max_tools} tools are supported.")
        return list(tools)

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits": self.policy.limits.to_dict(),
            "max_request_bytes": self.policy.max_request_bytes,
            "max_tools": self.policy.max_tools,
            "max_stop_sequences": self.policy.max_stop_sequences,
        }
