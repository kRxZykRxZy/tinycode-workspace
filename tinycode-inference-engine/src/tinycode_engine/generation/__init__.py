"""Token generation."""

from __future__ import annotations

from .generator import GenerationRequest, GenerationResult, Generator, TokenChunk
from .greedy import select_greedy
from .logits_processing import LogitsProcessors, apply_logits_processors
from .repetition import (
    apply_frequency_penalty,
    apply_presence_penalty,
    apply_repetition_penalty,
)
from .sampling import apply_temperature, sample_token, top_k_filter, top_p_filter
from .stopping import StopCondition, StopController, StopReason
from .stream import SSEWriter, chat_chunk, completion_chunk, format_sse

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "Generator",
    "LogitsProcessors",
    "SSEWriter",
    "StopCondition",
    "StopController",
    "StopReason",
    "TokenChunk",
    "apply_frequency_penalty",
    "apply_logits_processors",
    "apply_presence_penalty",
    "apply_repetition_penalty",
    "apply_temperature",
    "chat_chunk",
    "completion_chunk",
    "format_sse",
    "sample_token",
    "select_greedy",
    "top_k_filter",
    "top_p_filter",
]
