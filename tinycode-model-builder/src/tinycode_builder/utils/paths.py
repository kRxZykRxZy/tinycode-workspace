"""Path helpers for locating repository directories at runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["project_root", "resolve_repo_root", "ensure_dir", "BuilderPaths"]


def resolve_repo_root(start: Path | None = None) -> Path:
    """Return the repository root (the directory containing ``main.py``)."""

    here = (start or Path(__file__).resolve()).resolve()
    for parent in [here, *here.parents]:
        if (parent / "main.py").is_file() and (parent / "src").is_dir():
            return parent
    return here.parents[3] if len(here.parents) >= 4 else here.parent


def project_root() -> Path:
    """Return the repository root of the model builder."""

    return resolve_repo_root()


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@dataclass(frozen=True)
class BuilderPaths:
    """Resolved filesystem layout for a builder run."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "BuilderPaths":
        return cls(root=Path(root).resolve() if root else project_root())

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def tokenizer_dir(self) -> Path:
        return self.root / "tokenizer"

    def ensure(self) -> "BuilderPaths":
        for directory in (
            self.runs_dir,
            self.artifacts_dir,
            self.data_dir,
            self.checkpoints_dir,
            self.reports_dir,
            self.tokenizer_dir,
        ):
            ensure_dir(directory)
        return self

    def run_dir(self, name: str) -> Path:
        return ensure_dir(self.runs_dir / name)
