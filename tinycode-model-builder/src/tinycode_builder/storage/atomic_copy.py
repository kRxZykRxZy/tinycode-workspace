"""Atomic copies between storage locations."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CopyResult", "atomic_copy", "file_digest", "verify_copy"]


@dataclass
class CopyResult:
    """Outcome of a copy operation."""

    source: Path
    destination: Path
    bytes_copied: int = 0
    verified: bool = False
    sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "bytes_copied": self.bytes_copied,
            "verified": self.verified,
            "sha256": self.sha256,
        }


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: str | Path, destination: str | Path, verify: bool = True) -> CopyResult:
    """Copy ``source`` to ``destination`` atomically.

    The copy is written to a temporary file in the destination directory and
    renamed once it is complete, so an interrupted copy never destroys an
    existing file.
    """

    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(destination.parent), prefix=destination.name + ".", suffix=".part", delete=False
    )
    try:
        with handle, source.open("rb") as src:
            shutil.copyfileobj(src, handle, length=1 << 20)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, destination)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:  # pragma: no cover
            pass
        raise

    result = CopyResult(
        source=source,
        destination=destination,
        bytes_copied=destination.stat().st_size,
    )
    if verify:
        result.sha256 = file_digest(destination)
        result.verified = result.sha256 == file_digest(source)
    return result


def verify_copy(source: str | Path, destination: str | Path) -> bool:
    """Return whether the destination matches the source byte for byte."""

    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or not destination.is_file():
        return False
    if source.stat().st_size != destination.stat().st_size:
        return False
    return file_digest(source) == file_digest(destination)
