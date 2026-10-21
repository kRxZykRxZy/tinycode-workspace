"""Serialisation of training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["CHECKPOINT_FORMAT_VERSION", "serialize_checkpoint", "deserialize_checkpoint"]

CHECKPOINT_FORMAT_VERSION = "1.0.0"


def serialize_checkpoint(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a checkpoint payload atomically.

    The payload is written with ``torch.save`` into a temporary file that is
    renamed afterwards, so the previous known-good checkpoint is never
    destroyed by a failed write.
    """

    import os
    import tempfile

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(target.parent), prefix=target.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle:
            torch.save(payload, handle.name)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:  # pragma: no cover
            pass
        raise
    return target


def deserialize_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload."""

    import torch

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {target}")
    try:
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - torch < 2.0
        payload = torch.load(target, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {target} does not contain a dictionary payload.")
    return payload
