"""Parsing and validation of the artifact manifest.

This is the engine's own implementation of the shared contract (see
``contracts/artifact-manifest.v1.schema.json``).  It is deliberately
independent from the model builder: the two repositories only share the
*specification*, not the code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.errors import ArtifactError
from ..utils.version import ContractVersion, version_tuple

__all__ = ["ArtifactManifest", "MANIFEST_FILENAME", "load_manifest", "parse_manifest"]

MANIFEST_FILENAME = "artifact.json"

SUPPORTED_ARCHITECTURES = ("tinycode-decoder-v1",)
SUPPORTED_CONTRACT_MAJOR_MINOR = (1, 0)


@dataclass
class ArtifactManifest:
    """A validated artifact manifest."""

    raw: dict[str, Any]
    path: Path | None = None

    # ------------------------------------------------------------------
    @property
    def format_version(self) -> str:
        return str(self.raw.get("artifact_format_version", ""))

    @property
    def model_name(self) -> str:
        return str((self.raw.get("model") or {}).get("name", "tinycode"))

    @property
    def model_version(self) -> str:
        return str((self.raw.get("model") or {}).get("version", "0.0.0"))

    @property
    def architecture_id(self) -> str:
        return str((self.raw.get("architecture") or {}).get("architecture_id", ""))

    @property
    def architecture(self) -> dict[str, Any]:
        return dict(self.raw.get("architecture") or {})

    @property
    def context_length(self) -> int:
        return int(self.architecture.get("context_length", 0))

    @property
    def vocab_size(self) -> int:
        return int(self.architecture.get("vocab_size", 0))

    @property
    def parameter_target(self) -> int:
        return int((self.raw.get("parameters") or {}).get("requested_target", 0))

    @property
    def actual_parameters(self) -> int:
        return int((self.raw.get("parameters") or {}).get("actual_trainable", 0))

    @property
    def tokenizer_fingerprint(self) -> str:
        return str((self.raw.get("tokenizer") or {}).get("fingerprint", ""))

    @property
    def weights(self) -> dict[str, Any]:
        return dict(self.raw.get("weights") or {})

    @property
    def controller(self) -> dict[str, Any]:
        return dict(self.raw.get("controller") or {"present": False})

    @property
    def required_operators(self) -> list[str]:
        return list(self.raw.get("required_operators") or [])

    @property
    def files(self) -> dict[str, Any]:
        return dict(self.raw.get("files") or {})

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


def parse_manifest(payload: dict[str, Any], path: Path | None = None) -> ArtifactManifest:
    """Create a manifest object from a decoded JSON payload."""

    if not isinstance(payload, dict):
        raise ArtifactError("The artifact manifest must be a JSON object.")
    return ArtifactManifest(raw=payload, path=path)


def load_manifest(directory: str | Path) -> ArtifactManifest:
    """Load and structurally validate the manifest of an artifact directory."""

    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ArtifactError(f"No artifact manifest found at {manifest_path}.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"The artifact manifest is not valid JSON: {exc}") from exc
    return parse_manifest(payload, manifest_path)
