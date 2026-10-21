"""The Responses API endpoint.

Only a documented subset of the Responses API is implemented: a single
input, a text answer and usage reporting.  Unsupported fields are
rejected with a clear error instead of being silently ignored.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..orchestration.request_pipeline import InferenceRequest
from ..protocol.errors import APIError
from ..protocol.request_validation import validate_max_tokens, validate_temperature, validate_top_p
from ..protocol.response_serialization import serialize_responses_api
from .dependencies import (
    client_identifier,
    get_api_keys,
    get_engine,
    get_rate_limiter,
    prepare_prompt,
    resolve_model,
)
from .schemas.embeddings import ResponsesRequest

__all__ = ["router", "responses"]

router = APIRouter(tags=["responses"])


@router.post("/v1/responses")
async def responses(
    payload: ResponsesRequest,
    request: Request,
    engine: Any = Depends(get_engine),
    api_keys: Any = Depends(get_api_keys),
    rate_limiter: Any = Depends(get_rate_limiter),
) -> Any:
    """Serve a minimal, OpenAI compatible Responses API request."""

    api_keys.authenticate(request.headers.get("authorization"))
    rate_limiter.enforce(client_identifier(request))

    model_id = resolve_model(engine, payload.model)
    if payload.store:
        raise APIError(
            "Storing responses is not implemented by this server.",
            "invalid_request_error",
            status_code=400,
        )

    if isinstance(payload.input, str):
        text = payload.input
    elif isinstance(payload.input, list) and payload.input and isinstance(payload.input[0], str):
        text = "\n".join(str(item) for item in payload.input)
    else:
        parts: list[str] = []
        for item in payload.input or []:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for entry in content:
                        if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                            parts.append(entry["text"])
        text = "\n".join(parts)

    if not text.strip():
        raise APIError("'input' must not be empty.", "invalid_request_error", status_code=400)

    if payload.instructions:
        text = f"{payload.instructions}\n\n{text}"

    max_tokens = validate_max_tokens(payload.max_output_tokens)
    temperature = validate_temperature(payload.temperature)
    top_p = validate_top_p(payload.top_p)

    prompt_ids, _prompt_tokens = prepare_prompt(engine, text, max_tokens)
    request_id = engine.next_request_id()

    inference_request = InferenceRequest(
        request_id=request_id,
        model_id=model_id,
        prompt=text,
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    result = engine.generate(inference_request)
    return serialize_responses_api(
        model_id=model_id,
        request_id=request_id,
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
