"""Local filesystem storage backend."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.atomic_io import atomic_write_bytes

__all__ = ["LocalStorage", "StorageError"]


class StorageError(RuntimeError):
    """Raised when a storage operation fails."""


@dataclass
class LocalStorage:
    """Stores artefacts on the local filesystem."""

    root: Path

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    def path(self, *parts: str) -> Path:
        """Return a path inside the storage root."""

        return self.root.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    def ensure_dir(self, *parts: str) -> Path:
        target = self.path(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_bytes(self, relative: str, payload: bytes) -> Path:
        """Write bytes atomically."""

        target = self.path(relative)
        try:
            return atomic_write_bytes(target, payload)
        except OSError as exc:
            raise StorageError(f"Failed to write {target}: {exc}") from exc

    def write_text(self, relative: str, text: str) -> Path:
        return self.write_bytes(relative, text.encode("utf-8"))

    def read_bytes(self, relative: str) -> bytes:
        target = self.path(relative)
        if not target.is_file():
            raise StorageError(f"File not found: {target}")
        return target.read_bytes()

    def read_text(self, relative: str) -> str:
        return self.read_bytes(relative).decode("utf-8")

    def list_dir(self, relative: str = ".") -> list[Path]:
        target = self.path(relative)
        if not target.is_dir():
            return []
        return sorted(target.iterdir())

    def copy_into(self, source: Path, relative: str) -> Path:
        """Copy a file into the storage root atomically."""

        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    def delete(self, relative: str) -> bool:
        target = self.path(relative)
        if target.is_file():
            target.unlink()
            return True
        return False

    def free_space_bytes(self) -> int:
        usage = shutil.disk_usage(str(self.root))
        return int(usage.free)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "local",
            "root": str(self.root),
            "free_bytes": self.free_space_bytes(),
        }
