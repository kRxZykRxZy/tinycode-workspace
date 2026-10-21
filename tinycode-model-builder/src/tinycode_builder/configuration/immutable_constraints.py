"""Immutable platform constraints.

The context length of the TinyCode platform is **permanently** 1024 tokens.
It is not configurable through YAML, environment variables, API request
fields, dashboard controls or hardware detection.  Every component that
needs to reason about sequence lengths imports the constants from this
module so that there is exactly one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CONTEXT_LENGTH",
    "FIXED_CONTEXT_LENGTH",
    "CONTEXT_TOKENS",
    "ContextLengthViolation",
    "assert_context_length",
    "ensure_context_length",
    "max_prompt_tokens",
    "split_token_budget",
    "is_context_length_override_attempt",
]


class ContextLengthViolation(ValueError):
    """Raised when a component attempts to use a context length other than 1024."""


#: The one and only supported context length, in tokens.
CONTEXT_LENGTH: int = 1024

#: Aliases kept for readability at call sites.
FIXED_CONTEXT_LENGTH: int = CONTEXT_LENGTH
CONTEXT_TOKENS: int = CONTEXT_LENGTH

#: Parameter names that must never be used to override the context length.
_FORBIDDEN_OVERRIDE_NAMES = frozenset(
    {
        "context_length",
        "context_size",
        "context_window",
        "max_context",
        "max_context_length",
        "max_position_embeddings",
        "n_positions",
        "seq_len",
        "sequence_length",
        "max_seq_len",
        "max_sequence_length",
        "window_size",
        "rope_scaling",
        "n_ctx",
    }
)


def is_context_length_override_attempt(mapping: dict) -> str | None:
    """Return the offending key if ``mapping`` tries to override the context length.

    The check is intentionally strict: any key that looks like a context
    length control is rejected rather than silently ignored.
    """

    if not mapping:
        return None
    for key in mapping:
        normalised = str(key).strip().lower().replace("-", "_")
        if normalised in _FORBIDDEN_OVERRIDE_NAMES:
            return str(key)
    return None


def assert_context_length(value: int) -> int:
    """Validate that ``value`` is the fixed context length.

    Raises:
        ContextLengthViolation: if ``value`` is not exactly 1024.
    """

    if value != CONTEXT_LENGTH:
        raise ContextLengthViolation(
            f"Context length is fixed at {CONTEXT_LENGTH} tokens, got {value}. "
            "Adaptive context allocation and longer context tiers are not supported."
        )
    return value


def ensure_context_length(value: int) -> int:
    """Alias of :func:`assert_context_length` kept for readability."""

    return assert_context_length(value)


def max_prompt_tokens(reserved_output_tokens: int = 0) -> int:
    """Maximum number of prompt tokens allowed for a given output reservation.

    The prompt budget and the output budget live in the same 1024 token
    window.  A request with a 900 token prompt and a 300 token completion
    does not fit, so the caller must reserve room for the completion.

    Args:
        reserved_output_tokens: number of tokens to keep free for generation.

    Returns:
        The number of tokens that may be used by the prompt.
    """

    if reserved_output_tokens < 0:
        raise ValueError("reserved_output_tokens must be >= 0")
    remaining = CONTEXT_LENGTH - reserved_output_tokens
    if remaining <= 0:
        raise ValueError(
            f"Reserved output of {reserved_output_tokens} tokens does not fit in the "
            f"fixed {CONTEXT_LENGTH} token context."
        )
    return remaining


def split_token_budget(prompt_tokens: int, requested_output_tokens: int) -> tuple[int, int]:
    """Split the fixed context window into prompt and output budgets.

    Args:
        prompt_tokens: number of tokens already consumed by the prompt.
        requested_output_tokens: number of tokens the caller would like.

    Returns:
        ``(prompt_tokens, output_tokens)`` where the sum never exceeds 1024.
    """

    if prompt_tokens < 0 or requested_output_tokens < 0:
        raise ValueError("token budgets must be non-negative")
    if prompt_tokens > CONTEXT_LENGTH:
        raise ContextLengthViolation(
            f"Prompt of {prompt_tokens} tokens exceeds the fixed {CONTEXT_LENGTH} token context."
        )
    allowed_output = CONTEXT_LENGTH - prompt_tokens
    output = min(requested_output_tokens, allowed_output)
    return prompt_tokens, output


@dataclass(frozen=True)
class ContextPolicy:
    """Describes how an over-long prompt is handled."""

    strategy: str = "truncate"
    reserve_output_tokens: int = 0

    def effective_prompt_limit(self) -> int:
        return max_prompt_tokens(self.reserve_output_tokens)
