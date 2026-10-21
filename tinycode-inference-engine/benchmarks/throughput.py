#!/usr/bin/env python
"""Measure the throughput of the inference engine.

Usage:
    python benchmarks/throughput.py --model ../artifacts/tinycode-latest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_engine.orchestration.request_pipeline import InferenceRequest  # noqa: E402
from tinycode_engine.runtime.engine import EngineConfig, InferenceEngine  # noqa: E402
from tinycode_engine.server.startup import start_engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the inference throughput.")
    parser.add_argument("--model", required=True, help="Path of the artifact directory.")
    parser.add_argument("--prompt", default="Write a Python function that adds two numbers.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    config = EngineConfig(model_paths=[args.model], default_model_alias="bench")
    engine = InferenceEngine(config)
    report = start_engine(config, engine)
    if not report.models:
        print("no model could be loaded", file=sys.stderr)
        return 1

    prompt_ids = engine.tokenize(args.prompt)
    latencies: list[float] = []
    tokens = 0
    started = time.perf_counter()
    for index in range(max(1, args.iterations)):
        request = InferenceRequest(
            request_id=f"throughput-{index}",
            model_id=report.models[0],
            prompt=args.prompt,
            prompt_ids=list(prompt_ids),
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        request_started = time.perf_counter()
        result = engine.generate(request)
        latencies.append((time.perf_counter() - request_started) * 1000.0)
        tokens += result.completion_tokens

    elapsed = time.perf_counter() - started
    payload = {
        "model": report.models[0],
        "backend": report.backend,
        "iterations": len(latencies),
        "tokens": tokens,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(tokens / elapsed, 2) if elapsed else 0.0,
        "latency_ms_mean": round(sum(latencies) / len(latencies), 3),
        "latency_ms_min": round(min(latencies), 3),
        "latency_ms_max": round(max(latencies), 3),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
