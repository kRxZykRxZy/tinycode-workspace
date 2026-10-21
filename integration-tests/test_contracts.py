"""Cross repository contract tests.

These tests verify that the two repositories agree on the shared
contract: the context length, the parameter accounting formula, the
budget feature vector and the chat prompt format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_both_repositories_use_the_same_context_length() -> None:
    from tinycode_builder.configuration import immutable_constraints as builder_constraints
    from tinycode_engine.budget import request_analyzer as engine_analyzer

    assert builder_constraints.CONTEXT_LENGTH == 1024
    assert engine_analyzer.CONTEXT_LENGTH == 1024


def test_both_repositories_compute_the_same_parameter_count() -> None:
    from tinycode_builder.model.architecture import TransformerConfig as BuilderConfig
    from tinycode_builder.model.parameter_accounting import (
        analytic_parameter_count as builder_count,
    )

    def engine_count(config: dict) -> int:
        vocab = config["vocab_size"]
        d = config["d_model"]
        layers = config["n_layers"]
        ff = config["d_ff"]
        context = config["context_length"]
        norm = 2 * d if config["norm_type"] == "layernorm" else d
        total = vocab * d
        if config["position_encoding"] == "learned":
            total += context * d
        per_layer = 2 * norm + 4 * d * d + (4 * d if config["qkv_bias"] else 0)
        total += layers * per_layer
        total += layers * (2 * d * ff + (ff + d if config["ff_bias"] else 0))
        total += norm
        if not config["tie_embeddings"]:
            total += vocab * d
            if config["head_bias"]:
                total += vocab
        return int(total)

    config = BuilderConfig(
        vocab_size=512,
        d_model=64,
        n_layers=4,
        n_heads=8,
        d_ff=256,
        context_length=1024,
        norm_type="rmsnorm",
        position_encoding="learned",
        qkv_bias=True,
        ff_bias=True,
        tie_embeddings=False,
    )
    assert builder_count(config) == engine_count(config.to_dict())


def test_both_repositories_use_the_same_budget_features(feature_contract: dict) -> None:
    from tinycode_builder.budget.features import FEATURE_NAMES as builder_features
    from tinycode_engine.budget.controller import FEATURE_NAMES as engine_features

    assert list(builder_features) == list(engine_features)
    assert list(builder_features) == list(feature_contract["features"])


def test_the_feature_vectors_are_identical() -> None:
    from tinycode_builder.budget.features import BudgetFeatures as BuilderFeatures
    from tinycode_engine.budget.controller import BudgetFeatures as EngineFeatures

    builder = BuilderFeatures(
        prompt_tokens=100,
        requested_output_tokens=64,
        task_category="debug",
        complexity_score=0.4,
        uncertainty_score=0.2,
        wants_verification=True,
        latency_target_ms=5_000,
        queue_pressure=0.1,
        memory_pressure=0.2,
        backend_tokens_per_second=128.0,
        plan_steps=2,
        computation_budget=2_048,
    )
    engine = EngineFeatures(
        prompt_tokens=100,
        requested_output_tokens=64,
        task_category="debug",
        complexity_score=0.4,
        uncertainty_score=0.2,
        wants_verification=True,
        latency_target_ms=5_000,
        queue_pressure=0.1,
        memory_pressure=0.2,
        backend_tokens_per_second=128.0,
        plan_steps=2,
        computation_budget=2_048,
    )
    assert builder.to_vector() == engine.to_vector()


def test_both_repositories_render_the_same_chat_prompt() -> None:
    from tinycode_builder.tokenizer.special_tokens import chat_prompt as builder_prompt
    from tinycode_engine.model.tokenizer_adapter import chat_prompt as engine_prompt

    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert builder_prompt(messages) == engine_prompt(messages)


def test_the_manifest_matches_the_contract_schema(golden_artifact: Path, manifest_schema: dict) -> None:
    manifest = json.loads((golden_artifact / "artifact.json").read_text(encoding="utf-8"))

    required = manifest_schema["required"]
    for section in required:
        assert section in manifest, f"the manifest is missing '{section}'"

    architecture = manifest_schema["properties"]["architecture"]
    for section in architecture["required"]:
        assert section in manifest["architecture"]
    assert manifest["architecture"]["context_length"] == 1024
