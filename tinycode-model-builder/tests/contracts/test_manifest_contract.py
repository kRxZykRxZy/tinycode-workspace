"""Tests for the artifact manifest contract (builder side)."""

from __future__ import annotations

import json
from pathlib import Path

from tinycode_builder.export.artifact_manifest import (
    ARTIFACT_FORMAT_VERSION,
    build_manifest,
    validate_manifest,
)


def _manifest(**overrides) -> dict:
    architecture = {
        "vocab_size": 256,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 64,
        "context_length": 1024,
        "norm_type": "rmsnorm",
        "activation": "silu",
        "position_encoding": "learned",
        "qkv_bias": True,
        "ff_bias": True,
        "head_bias": False,
        "tie_embeddings": False,
        "architecture_id": "tinycode-decoder-v1",
    }
    payload = build_manifest(
        model_name="tinycode",
        model_version="0.1.0",
        architecture=architecture,
        parameter_target=1234,
        actual_parameters=1234,
        tokenizer_info={"fingerprint": "sha256:abc", "vocab_size": 256, "files": {}},
        weights={"format": "safetensors", "file": "model.safetensors", "sha256": "sha256:def"},
    )
    payload.update(overrides)
    return payload


def test_a_valid_manifest_has_no_problems() -> None:
    assert validate_manifest(_manifest()) == []


def test_manifest_version_is_pinned() -> None:
    manifest = _manifest()
    assert manifest["artifact_format_version"] == ARTIFACT_FORMAT_VERSION


def test_parameter_mismatch_is_reported() -> None:
    manifest = _manifest()
    manifest["parameters"]["actual_trainable"] = 999
    problems = validate_manifest(manifest)
    assert any("do not match" in problem for problem in problems)


def test_context_length_override_is_reported() -> None:
    manifest = _manifest()
    manifest["architecture"]["context_length"] = 2048
    problems = validate_manifest(manifest)
    assert any("context_length" in problem for problem in problems)


def test_missing_tokenizer_fingerprint_is_reported() -> None:
    manifest = _manifest()
    manifest["tokenizer"]["fingerprint"] = ""
    problems = validate_manifest(manifest)
    assert any("fingerprint" in problem for problem in problems)


def test_controller_reference_must_exist() -> None:
    manifest = _manifest()
    manifest["controller"] = {"present": True, "file": None}
    problems = validate_manifest(manifest)
    assert any("controller" in problem for problem in problems)


def test_manifest_is_json_serialisable() -> None:
    payload = json.dumps(_manifest())
    assert "artifact_format_version" in payload
