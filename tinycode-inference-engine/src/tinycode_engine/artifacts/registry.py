"""A registry of the artifacts that are loaded into the engine."""

from __future__ import annotations

import threading
from typing import Any

from ..utils.errors import ArtifactError
from .loader import LoadedArtifact, load_artifact

__all__ = ["ArtifactRegistry"]


class ArtifactRegistry:
    """Keeps the loaded artifacts addressable by their model id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._artifacts: dict[str, LoadedArtifact] = {}

    def load(self, directory: str, alias: str | None = None, **kwargs: Any) -> LoadedArtifact:
        """Load an artifact and register it."""

        artifact = load_artifact(directory, **kwargs)
        model_id = alias or artifact.model_id
        with self._lock:
            self._artifacts[model_id] = artifact
        return artifact

    def get(self, model_id: str) -> LoadedArtifact:
        """Return a registered artifact."""

        with self._lock:
            if model_id not in self._artifacts:
                raise ArtifactError(f"Unknown model: {model_id!r}")
            return self._artifacts[model_id]

    def has(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._artifacts

    def list(self) -> list[LoadedArtifact]:
        with self._lock:
            return list(self._artifacts.values())

    def model_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._artifacts)

    def default_model_id(self) -> str | None:
        with self._lock:
            if not self._artifacts:
                return None
            return sorted(self._artifacts)[0]

    def unload(self, model_id: str) -> None:
        with self._lock:
            self._artifacts.pop(model_id, None)
