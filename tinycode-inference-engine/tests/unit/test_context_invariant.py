"""The context length is fixed at 1024 tokens everywhere."""

from __future__ import annotations

import numpy as np
import pytest

from tinycode_engine.cache.kv_cache import KVCache
from tinycode_engine.model.model_adapter import TinyCodeModel
from tinycode_engine.protocol.request_validation import validate_max_tokens
from tinycode_engine.utils.errors import ModelError

from fixtures.artifact_factory import build_weights


def _model(context_length: int = 1024) -> TinyCodeModel:
    weights = build_weights(context_length=context_length)
    config = {
        "vocab_size": len(weights["embeddings.token.weight"]),
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 32,
        "context_length": context_length,
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "activation": "silu",
        "position_encoding": "learned",
        "qkv_bias": True,
        "ff_bias": True,
        "head_bias": False,
        "tie_embeddings": False,
    }
    return TinyCodeModel(config, weights)


def test_the_context_length_is_1024() -> None:
    assert _model().capabilities().context_length == 1024


def test_a_prompt_of_1024_tokens_fits() -> None:
    model = _model()
    token_ids = np.arange(1024, dtype=np.int64) % model.vocab_size
    logits = model.forward(token_ids)
    assert logits.shape == (model.vocab_size,)


def test_a_prompt_longer_than_the_context_is_rejected() -> None:
    model = _model()
    with pytest.raises(ModelError):
        model.forward(np.arange(1025, dtype=np.int64))


def test_the_cache_refuses_to_grow_beyond_the_context() -> None:
    cache = KVCache(n_layers=1, n_heads=2, head_dim=4, context_length=8)
    assert cache.can_append(8)
    assert not cache.can_append(9)
    cache.advance(8)
    assert cache.full()
    with pytest.raises(RuntimeError):
        cache.advance(1)


def test_max_tokens_cannot_exceed_the_context_length() -> None:
    from tinycode_engine.utils.errors import RequestError

    assert validate_max_tokens(1024) == 1024
    with pytest.raises(RequestError):
        validate_max_tokens(1025)
    with pytest.raises(RequestError):
        validate_max_tokens(0)


def test_the_generation_request_cannot_exceed_the_context() -> None:
    from tinycode_engine.generation.generator import GenerationRequest, Generator

    model = _model()
    generator = Generator(model, tokenizer=None)
    request = GenerationRequest(prompt_ids=list(range(1024)), max_tokens=64)
    with pytest.raises(ModelError):
        generator.generate(request)
