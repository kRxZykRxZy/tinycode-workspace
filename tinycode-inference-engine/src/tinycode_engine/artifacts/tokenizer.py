"""Tokenizer loading and verification for an artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..model.tokenizer_adapter import TokenizerAdapter
from ..utils.errors import ArtifactError
from .integrity import file_digest

__all__ = ["load_artifact_tokenizer", "verify_tokenizer_fingerprint"]


def verify_tokenizer_fingerprint(path: str | Path, expected: str) -> bool:
    """Check whether a tokenizer file matches the expected fingerprint."""

    if not expected.startswith("sha256:"):
        return False
    return f"sha256:{file_digest(path)}" == expected


def load_artifact_tokenizer(
    directory: str | Path, manifest: Any
) -> TokenizerAdapter:
    """Load the tokenizer of an artifact and verify its fingerprint."""

    directory = Path(directory)
    tokenizer_info = manifest.raw.get("tokenizer") if hasattr(manifest, "raw") else manifest.get("tokenizer", {})
    tokenizer_info = dict(tokenizer_info or {})

    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise ArtifactError(f"The artifact does not contain a tokenizer: {tokenizer_path}")

    expected = str(tokenizer_info.get("fingerprint") or "")
    if expected and not verify_tokenizer_fingerprint(tokenizer_path, expected):
        raise ArtifactError(
            "The tokenizer of the artifact does not match the fingerprint declared in "
            "the manifest; the artifact was not loaded."
        )

    vocabulary = tokenizer_info.get("vocab_size")
    adapter = TokenizerAdapter.from_file(tokenizer_path, fingerprint=expected)
    if isinstance(vocabulary, int) and vocabulary and adapter.vocab_size != vocabulary:
        raise ArtifactError(
            f"The tokenizer has {adapter.vocab_size} tokens but the manifest declares "
            f"{vocabulary}."
        )
    return adapter
