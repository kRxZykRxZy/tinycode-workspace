"""The adaptive budget layer works without a trained controller."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_the_baseline_controller_is_always_available() -> None:
    from tinycode_engine.budget.controller import BaselineBudgetController, BudgetFeatures

    controller = BaselineBudgetController()
    decision = controller.decide(BudgetFeatures(complexity_score=0.6), 256)
    assert decision.source == "baseline"
    assert decision.generation_tokens > 0


def test_an_incompatible_controller_falls_back_to_the_baseline() -> None:
    from tinycode_engine.budget.controller import BudgetFeatures, TrainedBudgetController

    controller = TrainedBudgetController({}, {"feature_schema_version": "budget_features.v0"})
    assert not controller.compatible
    decision = controller.decide(BudgetFeatures(complexity_score=0.6), 256)
    assert decision.source == "baseline"
    assert "baseline" in decision.rationale


def test_the_engine_serves_without_a_controller(tmp_path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    from tinycode_engine.artifacts.loader import load_artifact
    from tinycode_engine.orchestration.request_pipeline import InferenceRequest
    from tinycode_engine.runtime.engine import EngineConfig, InferenceEngine

    weights = {
        "embeddings.token.weight": np.zeros((32, 8), dtype=np.float32),
        "embeddings.position.weight": np.zeros((1024, 8), dtype=np.float32),
        "final_norm.weight": np.ones(8, dtype=np.float32),
        "lm_head.lm_head.weight": np.zeros((32, 8), dtype=np.float32),
    }
    for layer in range(1):
        prefix = f"layers.{layer}"
        weights[f"{prefix}.norm1.weight"] = np.ones(8, dtype=np.float32)
        weights[f"{prefix}.norm2.weight"] = np.ones(8, dtype=np.float32)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{prefix}.attention.{projection}.weight"] = np.zeros((8, 8), dtype=np.float32)
        weights[f"{prefix}.feed_forward.weight"] = np.zeros((16, 8), dtype=np.float32)
        weights[f"{prefix}.feed_forward_out.weight"] = np.zeros((8, 16), dtype=np.float32)

    directory = tmp_path / "artifact"
    directory.mkdir()
    save_file(weights, str(directory / "model.safetensors"))

    from tokenizers import Tokenizer, models, pre_tokenizers

    tokens = [f"tok{index}" for index in range(32)]
    tokenizer = Tokenizer(models.WordLevel(vocab={token: index for index, token in enumerate(tokens)}, unk_token="tok0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))

    import json

    from tinycode_engine.artifacts.integrity import file_digest

    manifest = {
        "artifact_format_version": "1.0.0",
        "model": {"name": "no-controller", "version": "0.1.0", "architecture_id": "tinycode-decoder-v1"},
        "architecture": {
            "vocab_size": 32,
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 16,
            "context_length": 1024,
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "activation": "silu",
            "position_encoding": "learned",
            "qkv_bias": False,
            "ff_bias": False,
            "head_bias": False,
            "tie_embeddings": False,
            "architecture_id": "tinycode-decoder-v1",
        },
        "parameters": {
            "requested_target": int(sum(array.size for array in weights.values())),
            "actual_trainable": int(sum(array.size for array in weights.values())),
            "verified": True,
        },
        "tokenizer": {
            "fingerprint": "sha256:" + file_digest(directory / "tokenizer.json"),
            "vocab_size": 32,
        },
        "weights": {
            "format": "safetensors",
            "file": "model.safetensors",
            "sha256": file_digest(directory / "model.safetensors"),
        },
        "controller": {"present": False},
        "required_operators": ["embedding", "linear", "rmsnorm", "causal_attention", "softmax", "silu"],
        "compatibility": {"min_engine_version": "0.1.0"},
        "files": {},
    }
    (directory / "artifact.json").write_text(json.dumps(manifest), encoding="utf-8")

    artifact = load_artifact(directory)
    assert artifact.controller is None

    config = EngineConfig(model_paths=[str(directory)], default_model_alias="no-controller")
    engine = InferenceEngine(config)
    engine.load_artifact(str(directory), alias="no-controller")
    request = InferenceRequest(
        request_id="baseline",
        model_id="no-controller",
        prompt="hello",
        prompt_ids=[1, 2, 3],
        max_tokens=4,
        temperature=0.0,
    )
    result = engine.generate(request)
    assert result.budget is not None
    assert result.budget.selected_policy in {"minimal", "standard", "extended", "verification"}
    engine.shutdown()
