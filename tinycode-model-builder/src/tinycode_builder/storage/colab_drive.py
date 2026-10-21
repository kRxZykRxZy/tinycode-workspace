"""Google Drive storage for Colab runs.

Only the Colab execution path imports ``google.colab``.  Outside Colab the
builder falls back to local storage and explains the limitation instead of
attempting a third party Drive mount.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ColabDriveStorage", "DriveUnavailableError", "is_colab_environment"]


class DriveUnavailableError(RuntimeError):
    """Raised when Google Drive cannot be used."""


def is_colab_environment() -> bool:
    """Return whether the current interpreter runs inside Google Colab."""

    try:
        import google.colab  # noqa: F401
    except Exception:
        return False
    return True


def _mount_drive(mount_point: str = "/content/drive") -> str:
    """Mount Google Drive using Colab's built-in module.

    Returns:
        The mount point that is ready to use.
    """

    if not is_colab_environment():
        raise DriveUnavailableError(
            "google.colab is not available. Google Drive mounting is only supported "
            "on Google Colab; outside Colab the builder uses local storage."
        )
    if os.path.exists(mount_point) and os.listdir(mount_point):
        return mount_point
    try:
        from google.colab import drive

        drive.mount(mount_point)
    except Exception as exc:
        raise DriveUnavailableError(f"Failed to mount Google Drive: {exc}") from exc
    return mount_point


@dataclass
class ColabDriveStorage:
    """Stores artefacts on Google Drive from a Colab runtime."""

    mount_point: str = "/content/drive"
    folder: str = "tinycode"
    _root: Path | None = None

    def __post_init__(self) -> None:
        self._root = None

    @property
    def root(self) -> Path:
        """The directory used for artefacts (mounting happens lazily)."""

        if self._root is None:
            mount_point = _mount_drive(self.mount_point)
            self._root = Path(mount_point) / "My Drive" / self.folder
            self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def ensure_dir(self, *parts: str) -> Path:
        target = self.path(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    def free_space_bytes(self) -> int:
        usage = os.statvfs(str(self.root))
        return int(usage.f_bavail * usage.f_frsize)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "google-drive",
            "mount_point": self.mount_point,
            "folder": self.folder,
            "available": is_colab_environment(),
        }
