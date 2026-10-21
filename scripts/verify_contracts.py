#!/usr/bin/env python
"""Verify that the shared contracts are consistent.

The script checks that

* the manifest schema matches the manifests that the builder emits,
* the feature contract matches the feature lists of both repositories,
* the golden fixture exists (and creates it when it is missing).

Usage:
    python scripts/verify_contracts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "tinycode-model-builder" / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "tinycode-inference-engine" / "src"))

CONTRACTS = WORKSPACE_ROOT / "contracts"


def main() -> int:
    problems: list[str] = []

    schema_path = CONTRACTS / "artifact-manifest.v1.schema.json"
    features_path = CONTRACTS / "budget-features.v1.json"
    if not schema_path.is_file():
        problems.append(f"missing {schema_path}")
    if not features_path.is_file():
        problems.append(f"missing {features_path}")
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 1

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    features = json.loads(features_path.read_text(encoding="utf-8"))

    from tinycode_builder.budget.features import FEATURE_NAMES as builder_features
    from tinycode_builder.export.artifact_manifest import (
        ARTIFACT_FORMAT_VERSION as builder_version,
    )
    from tinycode_engine.budget.controller import FEATURE_NAMES as engine_features
    from tinycode_engine.utils.version import ContractVersion

    if list(builder_features) != list(features["features"]):
        problems.append("the builder feature list does not match the contract")
    if list(engine_features) != list(features["features"]):
        problems.append("the engine feature list does not match the contract")
    if not ContractVersion().supports(builder_version):
        problems.append("the engine does not support the contract version of the builder")

    golden = CONTRACTS / "golden" / "tinycode-golden"
    if not (golden / "artifact.json").is_file():
        sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))
        import make_golden_fixture

        make_golden_fixture.main()
    manifest = json.loads((golden / "artifact.json").read_text(encoding="utf-8"))
    for section in schema["required"]:
        if section not in manifest:
            problems.append(f"the golden manifest is missing '{section}'")
    if manifest["architecture"]["context_length"] != 1024:
        problems.append("the golden manifest does not use the fixed context length")

    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 1

    print("the contracts are consistent")
    print(f"  contract version : {builder_version}")
    print(f"  feature schema   : {features['version']} ({len(features['features'])} features)")
    print(f"  golden artifact  : {golden}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
