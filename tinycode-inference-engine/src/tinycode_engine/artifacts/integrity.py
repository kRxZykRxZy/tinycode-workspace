"""Integrity verification of the artifact files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["IntegrityResult", "file_digest", "verify_files", "verify_manifest_files"]


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class IntegrityResult:
    """The result of an integrity check."""

    ok: bool = True
    checked: int = 0
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "missing": list(self.missing),
            "mismatched": list(self.mismatched),
        }


def verify_files(directory: str | Path, expected: dict[str, str]) -> IntegrityResult:
    """Verify that the files in ``directory`` match the expected digests."""

    directory = Path(directory)
    result = IntegrityResult()
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file():
            result.missing.append(name)
            result.ok = False
            continue
        result.checked += 1
        if file_digest(path) != digest:
            result.mismatched.append(name)
            result.ok = False
    return result


def verify_manifest_files(directory: str | Path, manifest: Any) -> IntegrityResult:
    """Verify the files declared by a manifest."""

    files = manifest.files if hasattr(manifest, "files") else dict(manifest.get("files", {}))
    expected = {
        name: str(meta.get("sha256"))
        for name, meta in files.items()
        if isinstance(meta, dict) and meta.get("sha256")
    }
    return verify_files(directory, expected)
