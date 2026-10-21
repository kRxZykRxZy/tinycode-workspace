"""Construction and validation of the versioned artifact manifest.

The manifest is the integration contract between the two repositories.  The
canonical schema lives in ``contracts/artifact-manifest.v1.schema.json``;
this module implements the builder side of that schema.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "REQUIRED_OPERATORS",
    "build_manifest",
    "validate_manifest",
    "ManifestError",
]

ARTIFACT_FORMAT_VERSION = "1.0.0"

#: Operators the inference engine must support to run the model.
REQUIRED_OPERATORS: tuple[str, ...] = (
    "embedding",
    "linear",
    "layernorm",
    "rmsnorm",
    "causal_attention",
    "softmax",
    "silu",
    "sampling",
)


class ManifestError(ValueError):
    """Raised when a manifest is malformed."""


def _file_entry(path: Path) -> dict[str, Any]:
    from ..checkpoint.integrity import file_digest

    return {
        "sha256": file_digest(path),
        "size": path.stat().st_size,
    }


def build_manifest(
    model_name: str,
    model_version: str,
    architecture: dict[str, Any],
    parameter_target: int,
    actual_parameters: int,
    tokenizer_info: dict[str, Any],
    weights: dict[str, Any],
    files: dict[str, Path] | None = None,
    controller: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    export: dict[str, Any] | None = None,
    compatibility: dict[str, Any] | None = None,
    precisions: tuple[str, ...] = ("float32",),
    parameter_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the artifact manifest dictionary."""

    manifest: dict[str, Any] = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_at": time.time(),
        "model": {
            "name": model_name,
            "version": model_version,
            "architecture_id": architecture.get("architecture_id", "tinycode-decoder-v1"),
        },
        "architecture": dict(architecture),
        "parameters": {
            "requested_target": int(parameter_target),
            "actual_trainable": int(actual_parameters),
            "verified": int(actual_parameters) == int(parameter_target),
            "breakdown": dict(parameter_breakdown or {}),
        },
        "tokenizer": dict(tokenizer_info),
        "weights": dict(weights),
        "precisions": list(precisions),
        "required_operators": list(REQUIRED_OPERATORS),
        "controller": dict(controller or {"present": False}),
        "training": dict(training or {}),
        "export": dict(export or {}),
        "compatibility": dict(
            compatibility
            or {
                "min_engine_version": "0.1.0",
                "min_contract_version": ARTIFACT_FORMAT_VERSION,
                "supported_backends": ["cpu", "cuda", "intel_gpu", "arm_neon"],
            }
        ),
        "files": {
            name: _file_entry(Path(path)) for name, path in (files or {}).items() if Path(path).is_file()
        },
    }
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return a list of problems found in ``manifest`` (empty when valid)."""

    problems: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]

    version = manifest.get("artifact_format_version")
    if not version:
        problems.append("artifact_format_version is missing")
    elif version != ARTIFACT_FORMAT_VERSION:
        problems.append(
            f"unsupported artifact_format_version {version!r} (expected {ARTIFACT_FORMAT_VERSION})"
        )

    for section in ("model", "architecture", "parameters", "tokenizer", "weights"):
        if section not in manifest:
            problems.append(f"section {section!r} is missing")

    architecture = manifest.get("architecture", {}) or {}
    if architecture.get("context_length") != 1024:
        problems.append(
            f"context_length must be 1024, got {architecture.get('context_length')!r}"
        )

    parameters = manifest.get("parameters", {}) or {}
    target = parameters.get("requested_target")
    actual = parameters.get("actual_trainable")
    if not isinstance(target, int) or not isinstance(actual, int):
        problems.append("parameter target and actual count must be integers")
    elif target != actual:
        problems.append(
            f"actual trainable parameters ({actual}) do not match the requested target ({target})"
        )
    if not parameters.get("verified", False):
        problems.append("parameter count was not verified")

    tokenizer = manifest.get("tokenizer", {}) or {}
    if not tokenizer.get("fingerprint"):
        problems.append("tokenizer fingerprint is missing")
    vocab = tokenizer.get("vocab_size")
    architecture_vocab = architecture.get("vocab_size")
    if isinstance(vocab, int) and isinstance(architecture_vocab, int) and vocab != architecture_vocab:
        problems.append(
            f"tokenizer vocab size ({vocab}) does not match the architecture ({architecture_vocab})"
        )

    weights = manifest.get("weights", {}) or {}
    if not weights.get("file"):
        problems.append("weights.file is missing")
    if not weights.get("sha256"):
        problems.append("weights.sha256 is missing")

    controller = manifest.get("controller") or {"present": False}
    if controller.get("present") and not controller.get("file"):
        problems.append("controller is marked present but no file is referenced")

    return problems
