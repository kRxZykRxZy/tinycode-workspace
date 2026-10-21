"""Storage backends for the model builder."""

from __future__ import annotations

from .atomic_copy import CopyResult, atomic_copy, file_digest, verify_copy
from .colab_drive import ColabDriveStorage, DriveUnavailableError, is_colab_environment
from .local import LocalStorage
from .storage_policy import StoragePolicy, resolve_storage
from .synchronization import SyncResult, Synchronizer, sync_file

__all__ = [
    "ColabDriveStorage",
    "CopyResult",
    "DriveUnavailableError",
    "LocalStorage",
    "StoragePolicy",
    "SyncResult",
    "Synchronizer",
    "atomic_copy",
    "file_digest",
    "is_colab_environment",
    "resolve_storage",
    "sync_file",
    "verify_copy",
]
