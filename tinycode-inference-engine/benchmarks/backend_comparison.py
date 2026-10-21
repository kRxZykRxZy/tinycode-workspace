#!/usr/bin/env python
"""Compare the available backends.

The script reports which backends *could* be initialised and how long
each of them needs to load the model.  It never claims acceleration for
a backend that could not be verified.

Usage:
    python benchmarks/backend_comparison.py --model ../artifacts/tinycode-latest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_engine.backends.registry import BackendRegistry  # noqa: E402
from tinycode_engine.runtime.engine import EngineConfig, InferenceEngine  # noqa: E402
from tinycode_engine.server.startup import start_engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the available backends.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    config = EngineConfig(model_paths=[args.model], default_model_alias="bench")
    engine = InferenceEngine(config)
    report = start_engine(config, engine)
    if not report.models:
        print("no model could be loaded", file=sys.stderr)
        return 1

    results = []
    registry = BackendRegistry({"preferred": "auto"})
    for backend in registry.candidates():
        entry: dict[str, object] = {"backend": backend.name}
        started = time.perf_counter()
        try:
            backend.initialize()
            capabilities = backend.capabilities()
            entry["available"] = True
            entry["accelerated"] = capabilities.is_accelerated
            entry["provider"] = capabilities.execution_provider
            entry["notes"] = capabilities.notes
        except Exception as exc:
            entry["available"] = False
            entry["reason"] = str(exc)
        entry["initialize_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        results.append(entry)
        try:
            backend.shutdown()
        except Exception:
            pass

    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
