"""Checkpoint integrity verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["IntegrityResult", "file_digest", "directory_digests", "verify_files"]


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digests(directory: str | Path, patterns: tuple[str, ...] = ("*",)) -> dict[str, str]:
    """Return ``{relative_name: sha256}`` for the files in a directory."""

    directory = Path(directory)
    digests: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path.is_file():
                digests[path.name] = file_digest(path)
    return digests


@dataclass
class IntegrityResult:
    """The outcome of an integrity check."""

    ok: bool = True
    checked: int = 0
    missing: list[str] = None  # type: ignore[assignment]
    mismatched: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.missing is None:
            self.missing = []
        if self.mismatched is None:
            self.mismatched = []

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
