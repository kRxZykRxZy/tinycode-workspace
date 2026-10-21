"""Selection of the storage backend for the current environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .colab_drive import ColabDriveStorage, DriveUnavailableError, is_colab_environment
from .local import LocalStorage

__all__ = ["StoragePolicy", "resolve_storage", "describe_storage"]


@dataclass
class StoragePolicy:
    """Describes where artefacts are written."""

    kind: str = "local"
    local_root: Path = Path(".")
    drive_folder: str = "tinycode"
    prefer_drive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "local_root": str(self.local_root),
            "drive_folder": self.drive_folder,
            "prefer_drive": self.prefer_drive,
            "colab": is_colab_environment(),
        }


def resolve_storage(policy: StoragePolicy) -> tuple[Any, str]:
    """Return ``(storage_backend, explanation)``.

    On Colab the builder writes to Google Drive (through Colab's own
    ``google.colab`` module).  Everywhere else the builder uses local
    storage and says so explicitly.
    """

    if policy.prefer_drive and is_colab_environment():
        try:
            backend = ColabDriveStorage(folder=policy.drive_folder)
            # Force the mount now so that failures are reported early.
            _ = backend.root
            return backend, "Google Drive (Colab runtime)"
        except DriveUnavailableError as exc:
            fallback = LocalStorage(policy.local_root)
            return fallback, f"Google Drive unavailable ({exc}); using local storage"
    backend = LocalStorage(policy.local_root)
    return backend, "Local filesystem (Drive mounting is only available on Google Colab)"


def describe_storage(policy: StoragePolicy) -> dict[str, Any]:
    """Return a JSON friendly description of the storage policy."""

    return policy.to_dict()
