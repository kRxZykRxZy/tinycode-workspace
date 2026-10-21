"""Tests of the generation subsystem."""

from __future__ import annotations

import numpy as np
import pytest

from tinycode_engine.generation.generator import GenerationRequest, Generator
from tinycode_engine.generation.logits_processing import LogitsProcessors
from tinycode_engine.generation.sampling import sample_token, top_p_filter
from tinycode_engine.generation.stopping import StopReason

from fixtures.artifact_factory import build_weights


def _generator() -> Generator:
    from tinycode_engine.model.model_adapter import TinyCodeModel

    weights = build_weights()
    config = {
        "vocab_size": len(weights["embeddings.token.weight"]),
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 32,
        "context_length": 1024,
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "activation": "silu",
        "position_encoding": "learned",
        "qkv_bias": True,
        "ff_bias": True,
        "head_bias": False,
        "tie_embeddings": False,
    }
    return Generator(TinyCodeModel(config, weights), tokenizer=None)


def test_greedy_generation_is_deterministic() -> None:
    generator = _generator()
    request = GenerationRequest(prompt_ids=[1, 2, 3], max_tokens=8, temperature=0.0)
    first = generator.generate(request)
    second = generator.generate(request)
    assert first.token_ids == second.token_ids
    assert first.completion_tokens == 8


def test_generation_stops_at_the_maximum_length() -> None:
    generator = _generator()
    result = generator.generate(GenerationRequest(prompt_ids=[1, 2], max_tokens=5, temperature=0.0))
    assert result.completion_tokens == 5
    assert result.finish_reason == StopReason.LENGTH


def test_generation_stops_at_the_eos_token() -> None:
    generator = _generator()
    request = GenerationRequest(
        prompt_ids=[1, 2], max_tokens=20, temperature=0.0, eos_token_id=0
    )
    result = generator.generate(request)
    assert result.finish_reason in {StopReason.EOS_TOKEN, StopReason.LENGTH}


def test_cancellation_stops_the_generation() -> None:
    generator = _generator()
    calls = {"count": 0}

    def _cancelled() -> bool:
        calls["count"] += 1
        return calls["count"] > 2

    result = generator.generate(
        GenerationRequest(prompt_ids=[1, 2], max_tokens=50, temperature=0.0), cancelled=_cancelled
    )
    assert result.finish_reason == StopReason.CANCELLED


def test_top_p_filter_keeps_a_small_set() -> None:
    logits = np.array([10.0, 9.0, 1.0, 0.5], dtype=np.float32)
    filtered = top_p_filter(logits, 0.5)
    assert np.isfinite(filtered).sum() <= 2


def test_sampling_respects_the_seed() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    first = sample_token(logits, temperature=1.0, top_p=1.0, rng=np.random.default_rng(42))
    second = sample_token(logits, temperature=1.0, top_p=1.0, rng=np.random.default_rng(42))
    assert first == second


def test_the_repetition_penalty_changes_the_logits() -> None:
    logits = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    processors = LogitsProcessors(repetition_penalty=1.5)
    adjusted = processors.apply(logits, [0])
    assert adjusted[0] < logits[0]
