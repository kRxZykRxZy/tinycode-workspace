"""Tests of the artifact manifest validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinycode_engine.artifacts.compatibility import check_compatibility
from tinycode_engine.artifacts.integrity import verify_manifest_files
from tinycode_engine.artifacts.loader import load_artifact
from tinycode_engine.artifacts.manifest import load_manifest
from tinycode_engine.utils.errors import ArtifactError


def test_a_valid_artifact_is_accepted(artifact_dir: Path) -> None:
    artifact = load_artifact(artifact_dir)
    assert artifact.model_id == "tinycode-test"
    assert artifact.manifest.context_length == 1024
    assert artifact.tokenizer is not None


def test_the_manifest_files_are_verified(artifact_dir: Path) -> None:
    manifest = load_manifest(artifact_dir)
    result = verify_manifest_files(artifact_dir, manifest)
    assert result.ok, result.to_dict()


def test_a_tampered_weight_file_is_rejected(artifact_dir: Path) -> None:
    weights = artifact_dir / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(ArtifactError):
        load_artifact(artifact_dir)


def test_a_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        load_artifact(tmp_path)


def test_a_context_length_override_is_rejected(artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "artifact.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["architecture"]["context_length"] = 2048
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError) as excinfo:
        load_artifact(artifact_dir)
    assert "1024" in str(excinfo.value)


def test_an_unsupported_architecture_is_rejected(artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "artifact.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["architecture"]["architecture_id"] = "some-other-model"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactError) as excinfo:
        load_artifact(artifact_dir)
    assert "architecture" in str(excinfo.value)


def test_a_parameter_mismatch_is_reported_as_a_warning(artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "artifact.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["parameters"]["actual_trainable"] = 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = load_artifact(artifact_dir, verify_integrity=False)
    assert any("parameter" in warning for warning in artifact.warnings)


def test_the_compatibility_report_lists_the_problems(artifact_dir: Path) -> None:
    manifest = load_manifest(artifact_dir)
    manifest.raw["architecture"]["context_length"] = 4096
    report = check_compatibility(manifest)
    assert not report.ok
    assert any("context length" in problem for problem in report.problems)


def test_a_missing_tokenizer_fingerprint_is_reported(artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "artifact.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["tokenizer"]["fingerprint"] = ""
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_manifest(artifact_dir)
    report = check_compatibility(manifest)
    assert not report.ok
    assert any("fingerprint" in problem for problem in report.problems)
