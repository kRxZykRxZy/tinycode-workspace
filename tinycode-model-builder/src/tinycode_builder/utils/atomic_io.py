"""Atomic file writing helpers.

Checkpoints and manifests must never be left half written: a crash during
``save`` has to leave the previous known-good file untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_bytes", "atomic_write_text", "atomic_write_json", "AtomicFileWriter"]


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write ``payload`` to ``path`` atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(target.parent), prefix=target.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        raise
    _fsync_dir(target.parent)
    return target


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Write text to ``path`` atomically."""

    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """Serialise ``data`` as JSON and write it atomically."""

    text = json.dumps(data, indent=indent, sort_keys=True, ensure_ascii=False)
    return atomic_write_text(path, text + "\n")


class AtomicFileWriter:
    """Context manager that writes to a temporary file and renames it on success."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._tmp: Path | None = None
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = Path(
            tempfile.mkstemp(dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp")[1]
        )
        self._handle = self._tmp.open("wb")
        return self._handle

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._handle is not None
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        if exc_type is None:
            os.replace(self._tmp, self.path)
            _fsync_dir(self.path.parent)
        else:
            try:
                self._tmp.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
        return False


def copy_file(source: str | Path, destination: str | Path, chunk_size: int = 1 << 20) -> Path:
    """Copy a file atomically (write to temp, then rename)."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(destination.parent), prefix=destination.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle, source.open("rb") as src:
            shutil.copyfileobj(src, handle, length=chunk_size)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, destination)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:  # pragma: no cover
            pass
        raise
    return destination
