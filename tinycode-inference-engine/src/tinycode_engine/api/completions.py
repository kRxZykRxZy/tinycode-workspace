"""The text completions endpoint."""

from __future__ import annotations

from typing import Any, Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

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
from ..protocol.response_serialization import serialize_completion
from ..protocol.streaming import completion_stream_events
from .dependencies import (
    client_identifier,
    get_api_keys,
    get_engine,
    get_rate_limiter,
    prepare_prompt,
    resolve_model,
)
from .schemas.completions import CompletionRequest

__all__ = ["router", "completions"]

router = APIRouter(tags=["completions"])


@router.post("/v1/completions")
async def completions(
    payload: CompletionRequest,
    request: Request,
    engine: Any = Depends(get_engine),
    api_keys: Any = Depends(get_api_keys),
    rate_limiter: Any = Depends(get_rate_limiter),
) -> Any:
    """Serve an OpenAI compatible text completion."""

    api_keys.authenticate(request.headers.get("authorization"))
    rate_limiter.enforce(client_identifier(request))

    model_id = resolve_model(engine, payload.model)
    if payload.logprobs is not None:
        raise APIError(
            "'logprobs' is not supported by this server.",
            "invalid_request_error",
            status_code=400,
        )

    max_tokens = validate_max_tokens(payload.max_tokens)
    temperature = validate_temperature(payload.temperature)
    top_p = validate_top_p(payload.top_p)
    stop = validate_stop(payload.stop)
    seed = validate_seed(payload.seed)
    presence, frequency, repetition = validate_penalties(
        payload.presence_penalty, payload.frequency_penalty, payload.repetition_penalty
    )
    stream = validate_stream(payload.stream)

    prompts = [payload.prompt] if isinstance(payload.prompt, str) else list(payload.prompt)
    if len(prompts) > 1:
        raise APIError(
            "Only a single prompt per request is supported by this server.",
            "invalid_request_error",
            status_code=400,
        )

    prompt_text = prompts[0]
    prompt_ids, _prompt_tokens = prepare_prompt(engine, prompt_text, max_tokens)
    request_id = engine.next_request_id()

    inference_request = InferenceRequest(
        request_id=request_id,
        model_id=model_id,
        prompt=prompt_text,
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
    )

    if stream:
        def _events() -> Iterator[str]:
            chunks = engine.stream(inference_request)
            yield from completion_stream_events(model_id, request_id, chunks)

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
        )

    result = engine.generate(inference_request)
    text = result.text
    if payload.echo:
        text = prompt_text + text
    return serialize_completion(
        model_id=model_id,
        request_id=request_id,
        text=text,
        finish_reason=result.finish_reason,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
