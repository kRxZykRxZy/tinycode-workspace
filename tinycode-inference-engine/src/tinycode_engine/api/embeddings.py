"""The embeddings endpoint.

The vectors are the mean pooled hidden states of the language model
backbone.  They are useful for similarity search but they are *not* a
dedicated retrieval embedding model; this limitation is documented in the
compatibility matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, Request

from ..protocol.errors import APIError
from ..protocol.response_serialization import serialize_embeddings
from .dependencies import get_api_keys, get_engine, get_rate_limiter, resolve_model
from .schemas.embeddings import EmbeddingsRequest

__all__ = ["router", "embeddings", "mean_pooled_embedding"]

router = APIRouter(tags=["embeddings"])


def mean_pooled_embedding(model: Any, token_ids: list[int]) -> list[float]:
    """Return the mean pooled final hidden state of a token sequence."""

    cache = model._allocate_cache()  # noqa: SLF001 - internal helper of the model
    try:
        hidden = model._embed(np.asarray(token_ids, dtype=np.int64))  # noqa: SLF001
        for layer in range(model.n_layers):
            prefix = f"layers.{layer}"
            normed = model._norm(  # noqa: SLF001
                hidden,
                model.weights[f"{prefix}.norm1.weight"],
                model.weights.get(f"{prefix}.norm1.bias"),
            )
            from ..model.attention import merge_heads, split_heads

            query = split_heads(model._linear(normed, f"{prefix}.attention.q_proj"), model.n_heads)  # noqa: SLF001
            key = split_heads(model._linear(normed, f"{prefix}.attention.k_proj"), model.n_heads)  # noqa: SLF001
            value = split_heads(model._linear(normed, f"{prefix}.attention.v_proj"), model.n_heads)  # noqa: SLF001
            cache.keys[layer][:, : len(token_ids), :] = key
            cache.values[layer][:, : len(token_ids), :] = value
            context = model.ops.attention_with_cache(  # noqa: SLF001
                query, cache.keys[layer], cache.values[layer], len(token_ids), model.scale
            )
            attended = model._linear(merge_heads(context), f"{prefix}.attention.o_proj")  # noqa: SLF001
            hidden = hidden + attended
            normed = model._norm(  # noqa: SLF001
                hidden,
                model.weights[f"{prefix}.norm2.weight"],
                model.weights.get(f"{prefix}.norm2.bias"),
            )
            inner = model._linear(normed, f"{prefix}.feed_forward")  # noqa: SLF001
            hidden = hidden + model._linear(model._activate(inner), f"{prefix}.feed_forward_out")  # noqa: SLF001
        hidden = model._norm(  # noqa: SLF001
            hidden,
            model.weights["final_norm.weight"],
            model.weights.get("final_norm.bias"),
        )
    finally:
        cache.release()

    pooled = hidden.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return [float(value) for value in pooled]


@router.post("/v1/embeddings")
async def embeddings(
    payload: EmbeddingsRequest,
    request: Request,
    engine: Any = Depends(get_engine),
    api_keys: Any = Depends(get_api_keys),
    rate_limiter: Any = Depends(get_rate_limiter),
) -> Any:
    """Serve an OpenAI compatible embeddings request."""

    api_keys.authenticate(request.headers.get("authorization"))
    rate_limiter.enforce("embeddings")

    model_id = resolve_model(engine, payload.model)
    if engine.model is None:
        raise APIError("No model is loaded.", "service_unavailable", status_code=503)

    inputs = [payload.input] if isinstance(payload.input, str) else list(payload.input)
    vectors: list[list[float]] = []
    total_tokens = 0
    for text in inputs:
        if not isinstance(text, str):
            raise APIError("'input' must contain strings.", "invalid_request_error", status_code=400)
        token_ids = engine.tokenizer.encode(text)[:1024]
        if not token_ids:
            token_ids = [engine.tokenizer.pad_token_id]
        total_tokens += len(token_ids)
        vector = mean_pooled_embedding(engine.model, token_ids)
        if payload.dimensions:
            vector = vector[: payload.dimensions]
        vectors.append(vector)

    return serialize_embeddings(model_id, vectors, prompt_tokens=total_tokens)
