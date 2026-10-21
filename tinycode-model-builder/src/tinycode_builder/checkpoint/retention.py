"""Retention policy for checkpoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["RetentionPolicy", "select_for_deletion", "apply_retention"]


@dataclass
class RetentionPolicy:
    """How many checkpoints to keep."""

    keep_last: int = 3
    keep_every: int = 0  # 0 disables the periodic keeper

    def to_dict(self) -> dict[str, int]:
        return {"keep_last": self.keep_last, "keep_every": self.keep_every}


_STEP_PATTERN = re.compile(r"step-(\d+)")


def _step_of(path: Path) -> int:
    match = _STEP_PATTERN.search(path.name)
    return int(match.group(1)) if match else -1


def select_for_deletion(paths: list[Path], policy: RetentionPolicy) -> list[Path]:
    """Return the checkpoints that should be removed."""

    if policy.keep_last <= 0 and policy.keep_every <= 0:
        return []
    ordered = sorted(paths, key=lambda path: (_step_of(path), path.name))
    keep: set[Path] = set()

    if policy.keep_last > 0:
        keep.update(ordered[-policy.keep_last :])
    if policy.keep_every > 0:
        for path in ordered:
            step = _step_of(path)
            if step >= 0 and step % policy.keep_every == 0:
                keep.add(path)

    return [path for path in ordered if path not in keep]


def apply_retention(directory: str | Path, policy: RetentionPolicy, pattern: str = "checkpoint-*.pt") -> list[Path]:
    """Delete the checkpoints that the retention policy no longer keeps."""

    directory = Path(directory)
    if not directory.is_dir():
        return []
    paths = [path for path in directory.glob(pattern) if path.is_file()]
    removed: list[Path] = []
    for path in select_for_deletion(paths, policy):
        try:
            path.unlink()
            removed.append(path)
        except OSError:  # pragma: no cover - permission problems
            continue
    return removed
