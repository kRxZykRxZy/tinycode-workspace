"""Synchronisation of checkpoints to a secondary storage location."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic_copy import CopyResult, atomic_copy, verify_copy

__all__ = ["SyncResult", "Synchronizer", "sync_file"]


@dataclass
class SyncResult:
    """Outcome of a synchronisation attempt."""

    ok: bool = False
    verified: bool = False
    source: str = ""
    destination: str = ""
    bytes_copied: int = 0
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verified": self.verified,
            "source": self.source,
            "destination": self.destination,
            "bytes_copied": self.bytes_copied,
            "error": self.error,
            "details": self.details,
        }


def sync_file(
    source: str | Path,
    destination: str | Path,
    verify: bool = True,
) -> SyncResult:
    """Copy ``source`` to ``destination`` and verify the copy.

    A failure never removes the local source: the local checkpoint is
    always preserved.
    """

    source = Path(source)
    destination = Path(destination)
    try:
        copy: CopyResult = atomic_copy(source, destination, verify=verify)
    except Exception as exc:
        return SyncResult(ok=False, verified=False, source=str(source), destination=str(destination), error=str(exc))
    verified = copy.verified if verify else True
    if verify and not verified:
        return SyncResult(
            ok=False,
            verified=False,
            source=str(source),
            destination=str(destination),
            bytes_copied=copy.bytes_copied,
            error="Copied file does not match the source digest.",
        )
    return SyncResult(
        ok=True,
        verified=verified,
        source=str(source),
        destination=str(destination),
        bytes_copied=copy.bytes_copied,
        details={"sha256": copy.sha256},
    )


class Synchronizer:
    """Keeps a mirror of the checkpoint directory on a secondary storage."""

    def __init__(self, destination_root: str | Path | None = None, verify: bool = True) -> None:
        self.destination_root = Path(destination_root) if destination_root else None
        self.verify = verify
        self.history: list[SyncResult] = []

    @property
    def available(self) -> bool:
        return self.destination_root is not None

    def sync(self, source: str | Path, relative_name: str | None = None) -> SyncResult:
        """Synchronise a single file."""

        if self.destination_root is None:
            return SyncResult(ok=False, verified=False, error="No secondary storage configured.")
        source = Path(source)
        name = relative_name or source.name
        destination = self.destination_root / name
        result = sync_file(source, destination, verify=self.verify)
        self.history.append(result)
        if len(self.history) > 100:
            del self.history[:50]
        return result

    def verify_existing(self, source: str | Path, relative_name: str | None = None) -> bool:
        """Verify that a previously synchronised file still matches."""

        if self.destination_root is None:
            return False
        source = Path(source)
        name = relative_name or source.name
        return verify_copy(source, self.destination_root / name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination_root": str(self.destination_root) if self.destination_root else None,
            "available": self.available,
            "history": [result.to_dict() for result in self.history[-10:]],
        }
