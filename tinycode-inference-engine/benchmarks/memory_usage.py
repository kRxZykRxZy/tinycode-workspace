#!/usr/bin/env python
"""Report the memory usage of the runtime.

Usage:
    python benchmarks/memory_usage.py --model ../artifacts/tinycode-latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_engine.runtime.engine import EngineConfig, InferenceEngine  # noqa: E402
from tinycode_engine.server.startup import start_engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Report the memory usage of the runtime.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    config = EngineConfig(model_paths=[args.model], default_model_alias="bench")
    engine = InferenceEngine(config)
    report = start_engine(config, engine)
    if not report.models:
        print("no model could be loaded", file=sys.stderr)
        return 1

    weight_bytes = int(
        sum(array.nbytes for array in engine.registry.get(report.models[0]).weights.values())
    )
    process_rss = None
    try:
        import psutil

        process_rss = int(psutil.Process().memory_info().rss)
    except Exception:
        pass

    payload = {
        "model": report.models[0],
        "backend": report.backend,
        "weight_bytes": weight_bytes,
        "weight_mb": round(weight_bytes / (1024**2), 3),
        "process_rss_bytes": process_rss,
        "process_rss_mb": None if process_rss is None else round(process_rss / (1024**2), 3),
        "kv_cache_bytes_per_token_estimate": None,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
