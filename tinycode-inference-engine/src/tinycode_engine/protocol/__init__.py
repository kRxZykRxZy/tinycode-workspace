"""The OpenAI compatibility protocol layer."""

from __future__ import annotations

from .errors import APIError, error_payload, error_response
from .openai_compat import (
    DEFAULT_MODEL_ID,
    chat_completion_payload,
    completion_payload,
    embeddings_payload,
    model_list_payload,
    model_payload,
    usage_payload,
)
from .request_validation import (
    validate_max_tokens,
    validate_model_name,
    validate_penalties,
    validate_seed,
    validate_stop,
    validate_stream,
    validate_temperature,
    validate_top_p,
)
from .response_serialization import (
    serialize_chat_completion,
    serialize_completion,
    serialize_embeddings,
    serialize_models,
    serialize_responses_api,
)
from .streaming import chat_stream_events, completion_stream_events, sse_event

__all__ = [
    "APIError",
    "DEFAULT_MODEL_ID",
    "chat_completion_payload",
    "chat_stream_events",
    "completion_payload",
    "completion_stream_events",
    "embeddings_payload",
    "error_payload",
    "error_response",
    "model_list_payload",
    "model_payload",
    "serialize_chat_completion",
    "serialize_completion",
    "serialize_embeddings",
    "serialize_models",
    "serialize_responses_api",
    "sse_event",
    "usage_payload",
    "validate_max_tokens",
    "validate_model_name",
    "validate_penalties",
    "validate_seed",
    "validate_stop",
    "validate_stream",
    "validate_temperature",
    "validate_top_p",
]
