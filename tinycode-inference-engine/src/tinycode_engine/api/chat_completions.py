"""The chat completions endpoint."""

from __future__ import annotations

from typing import Any, Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..model.tokenizer_adapter import chat_prompt
from ..orchestration.request_pipeline import InferenceRequest
from ..protocol.errors import APIError
from ..protocol.request_validation import (
    validate_max_tokens,
    validate_penalties,
    validate_seed,
    validate_stop,
    validate_stream,
    validate_temperature,
    validate_top_p,
)
from ..protocol.response_serialization import serialize_chat_completion
from ..protocol.streaming import chat_stream_events
from .dependencies import (
    client_identifier,
    get_api_keys,
    get_engine,
    get_limits,
    get_rate_limiter,
    prepare_prompt,
    resolve_model,
)
from .schemas.completions import ChatCompletionRequest

__all__ = ["router", "chat_completions"]

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    engine: Any = Depends(get_engine),
    api_keys: Any = Depends(get_api_keys),
    rate_limiter: Any = Depends(get_rate_limiter),
    limits: Any = Depends(get_limits),
) -> Any:
    """Serve an OpenAI compatible chat completion."""

    api_keys.authenticate(request.headers.get("authorization"))
    rate_limiter.enforce(client_identifier(request))

    model_id = resolve_model(engine, payload.model)

    if payload.tools:
        raise APIError(
            "Tool calling is not implemented by this server; the request was rejected "
            "instead of silently ignoring the tools.",
            "invalid_request_error",
            status_code=400,
        )
    if payload.tool_choice is not None:
        raise APIError(
            "'tool_choice' is not supported by this server.",
            "invalid_request_error",
            status_code=400,
        )
    if payload.logprobs:
        raise APIError(
            "'logprobs' is not supported by this server.",
            "invalid_request_error",
            status_code=400,
        )
    if payload.response_format is not None and payload.response_format.type not in {
        "text",
        "json_object",
    }:
        raise APIError(
            f"Unsupported response_format type: {payload.response_format.type!r}.",
            "invalid_request_error",
            status_code=400,
        )

    max_tokens = validate_max_tokens(payload.effective_max_tokens())
    temperature = validate_temperature(payload.temperature)
    top_p = validate_top_p(payload.top_p)
    stop = validate_stop(payload.stop)
    seed = validate_seed(payload.seed)
    presence, frequency, repetition = validate_penalties(
        payload.presence_penalty, payload.frequency_penalty, payload.repetition_penalty
    )
    stream = validate_stream(payload.stream)

    messages = [{"role": message.role, "content": message.content or ""} for message in payload.messages]
    prompt = chat_prompt(messages)
    prompt_ids, prompt_tokens = prepare_prompt(engine, prompt, max_tokens)

    request_id = engine.next_request_id()
    inference_request = InferenceRequest(
        request_id=request_id,
        model_id=model_id,
        prompt=prompt,
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=payload.top_k or 0,
        stop=stop,
        seed=seed,
        repetition_penalty=repetition,
        frequency_penalty=frequency,
        presence_penalty=presence,
        response_format=(
            payload.response_format.model_dump() if payload.response_format is not None else None
        ),
        tools=[],
    )

    if stream:
        def _events() -> Iterator[str]:
            chunks = engine.stream(inference_request)
            yield from chat_stream_events(model_id, request_id, chunks)

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
        )

    result = engine.generate(inference_request)
    return serialize_chat_completion(
        model_id=model_id,
        request_id=request_id,
        content=result.text,
        finish_reason=result.finish_reason,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
