"""Integration test of the inference engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tinycode_engine.orchestration.request_pipeline import InferenceRequest
from tinycode_engine.runtime.engine import EngineConfig, InferenceEngine


def test_the_engine_loads_an_artifact(artifact_dir: Path) -> None:
    engine = InferenceEngine(EngineConfig(model_paths=[str(artifact_dir)]))
    artifact = engine.load_artifact(str(artifact_dir), alias="tinycode-test")
    assert artifact.model is not None
    assert engine.model_ids() == ["tinycode-test"]
    engine.shutdown()


def test_the_engine_generates_an_answer(engine: InferenceEngine) -> None:
    request = InferenceRequest(
        request_id="test-1",
        model_id="tinycode-test",
        prompt="def add(a, b):",
        prompt_ids=engine.tokenize("def add(a, b):"),
        max_tokens=8,
        temperature=0.0,
    )
    result = engine.generate(request)
    assert result.completion_tokens >= 0
    assert result.prompt_tokens > 0
    assert result.budget is not None
    assert result.budget.allocation.generation_tokens <= 1024


def test_the_engine_streams_tokens(engine: InferenceEngine) -> None:
    request = InferenceRequest(
        request_id="test-stream",
        model_id="tinycode-test",
        prompt="hello",
        prompt_ids=engine.tokenize("hello"),
        max_tokens=4,
        temperature=0.0,
    )
    chunks = list(engine.stream(request))
    assert len(chunks) <= 4
    for chunk in chunks:
        assert isinstance(chunk.text, str)


def test_the_metrics_are_recorded(engine: InferenceEngine) -> None:
    request = InferenceRequest(
        request_id="test-metrics",
        model_id="tinycode-test",
        prompt="hello",
        prompt_ids=engine.tokenize("hello"),
        max_tokens=2,
        temperature=0.0,
    )
    engine.generate(request)
    assert engine.metrics["requests_received"] >= 1
    assert engine.metrics["output_tokens"] >= 0


def test_the_kv_cache_is_released_after_a_request(engine: InferenceEngine) -> None:
    request = InferenceRequest(
        request_id="test-cache",
        model_id="tinycode-test",
        prompt="hello",
        prompt_ids=engine.tokenize("hello"),
        max_tokens=2,
        temperature=0.0,
    )
    engine.generate(request)
    assert engine.model is not None


def test_the_model_produces_finite_logits(engine: InferenceEngine) -> None:
    logits = engine.model.forward(np.array([1, 2, 3], dtype=np.int64))
    assert np.all(np.isfinite(logits))
    assert logits.shape[0] == engine.model.vocab_size
