"""Shared pytest fixtures for the inference engine test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for path in (str(SRC), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fixtures.artifact_factory import build_artifact  # noqa: E402
from tinycode_engine.runtime.engine import EngineConfig, InferenceEngine  # noqa: E402


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    """Create a small artifact directory."""

    return build_artifact(tmp_path / "artifact")


@pytest.fixture()
def engine(artifact_dir: Path):
    """Create an engine with the small artifact loaded."""

    config = EngineConfig(
        model_paths=[str(artifact_dir)],
        default_model_alias="tinycode-test",
        api_keys=["test-key"],
        require_api_key=True,
    )
    instance = InferenceEngine(config)
    instance.load_artifact(str(artifact_dir), alias="tinycode-test")
    yield instance
    instance.shutdown()


@pytest.fixture()
def client(engine: InferenceEngine):
    """Return a test client for the HTTP API."""

    from fastapi.testclient import TestClient

    from tinycode_engine.api.app import create_app

    app = create_app(engine, engine.config)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Return the default authentication headers."""

    return {"Authorization": "Bearer test-key"}
