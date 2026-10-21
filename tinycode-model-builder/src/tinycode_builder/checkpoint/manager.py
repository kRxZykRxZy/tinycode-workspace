"""Checkpoint manager: save, verify, list and resume."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.atomic_io import atomic_write_json
from ..utils.errors import CheckpointError
from .integrity import file_digest, verify_files
from .manifest import CheckpointManifest
from .retention import RetentionPolicy, apply_retention
from .serializer import deserialize_checkpoint, serialize_checkpoint

__all__ = ["CheckpointManager", "CheckpointInfo"]


@dataclass
class CheckpointInfo:
    """Summary of a stored checkpoint."""

    path: Path
    step: int = 0
    committed_training_tokens: int = 0
    size_bytes: int = 0
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "step": self.step,
            "committed_training_tokens": self.committed_training_tokens,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


class CheckpointManager:
    """Owns the checkpoint directory."""

    def __init__(
        self,
        directory: str | Path,
        retention: RetentionPolicy | None = None,
        verify_integrity: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retention = retention or RetentionPolicy()
        self.verify_integrity = verify_integrity

    # ------------------------------------------------------------------
    # naming
    # ------------------------------------------------------------------
    def path_for(self, step: int) -> Path:
        return self.directory / f"checkpoint-{step:08d}.pt"

    def manifest_path_for(self, step: int) -> Path:
        return self.directory / f"checkpoint-{step:08d}.manifest.json"

    def latest(self) -> Path | None:
        """Return the newest checkpoint file, if any."""

        candidates = sorted(self.directory.glob("checkpoint-*.pt"))
        return candidates[-1] if candidates else None

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """Return information about every stored checkpoint."""

        infos: list[CheckpointInfo] = []
        for path in sorted(self.directory.glob("checkpoint-*.pt")):
            info = CheckpointInfo(
                path=path,
                size_bytes=path.stat().st_size,
                sha256=file_digest(path) if self.verify_integrity else "",
            )
            manifest = self._read_manifest(path)
            if manifest is not None:
                info.step = manifest.step
                info.committed_training_tokens = manifest.committed_training_tokens
            infos.append(info)
        return infos

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------
    def save(
        self,
        payload: dict[str, Any],
        step: int,
        manifest: CheckpointManifest | None = None,
        sync: Any = None,
    ) -> CheckpointInfo:
        """Persist a checkpoint payload and its manifest.

        The previous known-good checkpoint is preserved: the new file is
        written atomically next to the old one.
        """

        payload = dict(payload)
        payload.setdefault("step", step)
        manifest = manifest or CheckpointManifest(step=step)
        manifest.step = step
        manifest.created_at = time.time()

        path = self.path_for(step)
        try:
            serialize_checkpoint(payload, path)
        except Exception as exc:
            raise CheckpointError(f"Failed to write checkpoint {path}: {exc}") from exc

        manifest.files = {path.name: {"sha256": file_digest(path), "size": path.stat().st_size}}
        atomic_write_json(self.manifest_path_for(step), manifest.to_dict())

        info = CheckpointInfo(
            path=path,
            step=step,
            committed_training_tokens=manifest.committed_training_tokens,
            size_bytes=path.stat().st_size,
            sha256=manifest.files[path.name]["sha256"],
        )

        if sync is not None and getattr(sync, "available", False):
            sync.sync(path, relative_name=path.name)

        apply_retention(self.directory, self.retention)
        return info

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load a checkpoint payload (the latest one by default)."""

        target = Path(path) if path is not None else self.latest()
        if target is None:
            raise CheckpointError("No checkpoint is available to resume from.")
        if self.verify_integrity:
            manifest = self._read_manifest(target)
            if manifest is not None:
                expected = manifest.files.get(target.name, {}).get("sha256")
                if expected:
                    result = verify_files(self.directory, {target.name: expected})
                    if not result.ok:
                        raise CheckpointError(
                            f"Checkpoint integrity check failed for {target}: "
                            f"missing={result.missing} mismatched={result.mismatched}"
                        )
        return deserialize_checkpoint(target)

    def verify(self, path: str | Path | None = None) -> dict[str, Any]:
        """Verify the integrity of a checkpoint against its manifest."""

        target = Path(path) if path is not None else self.latest()
        if target is None:
            return {"ok": False, "reason": "no checkpoint"}
        manifest = self._read_manifest(target)
        if manifest is None:
            return {"ok": False, "reason": "no manifest"}
        expected = {name: meta["sha256"] for name, meta in manifest.files.items() if "sha256" in meta}
        result = verify_files(self.directory, expected)
        return result.to_dict()

    # ------------------------------------------------------------------
    def _read_manifest(self, checkpoint_path: Path) -> CheckpointManifest | None:
        manifest_path = checkpoint_path.with_suffix("").with_suffix(".manifest.json")
        if not manifest_path.is_file():
            manifest_path = self.directory / (checkpoint_path.stem + ".manifest.json")
        if not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return CheckpointManifest.from_dict(payload)
