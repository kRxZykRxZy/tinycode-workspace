"""Fixtures of the cross repository integration tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BUILDER_SRC = WORKSPACE_ROOT / "tinycode-model-builder" / "src"
ENGINE_SRC = WORKSPACE_ROOT / "tinycode-inference-engine" / "src"

for path in (str(BUILDER_SRC), str(ENGINE_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def golden_artifact(tmp_path_factory) -> Path:
    """Return the golden artifact, creating it when it does not exist."""

    golden = WORKSPACE_ROOT / "contracts" / "golden" / "tinycode-golden"
    if not (golden / "artifact.json").is_file():
        sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))
        import make_golden_fixture

        make_golden_fixture.main()
    return golden


@pytest.fixture(scope="session")
def workspace_root() -> Path:
    return WORKSPACE_ROOT


@pytest.fixture(scope="session")
def manifest_schema() -> dict:
    """Return the canonical manifest schema."""

    path = WORKSPACE_ROOT / "contracts" / "artifact-manifest.v1.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def feature_contract() -> dict:
    """Return the canonical budget feature contract."""

    path = WORKSPACE_ROOT / "contracts" / "budget-features.v1.json"
    return json.loads(path.read_text(encoding="utf-8"))
