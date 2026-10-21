#!/usr/bin/env python
"""Measure the latency of the inference engine (time to first token included).

Usage:
    python benchmarks/latency.py --model ../artifacts/tinycode-latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_engine.orchestration.request_pipeline import InferenceRequest  # noqa: E402
from tinycode_engine.runtime.engine import EngineConfig, InferenceEngine  # noqa: E402
from tinycode_engine.server.startup import start_engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the inference latency.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain what a Python decorator is.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    config = EngineConfig(model_paths=[args.model], default_model_alias="bench")
    engine = InferenceEngine(config)
    report = start_engine(config, engine)
    if not report.models:
        print("no model could be loaded", file=sys.stderr)
        return 1

    prompt_ids = engine.tokenize(args.prompt)
    totals: list[float] = []
    firsts: list[float] = []
    for index in range(max(1, args.iterations)):
        request = InferenceRequest(
            request_id=f"latency-{index}",
            model_id=report.models[0],
            prompt=args.prompt,
            prompt_ids=list(prompt_ids),
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        result = engine.generate(request)
        totals.append(result.elapsed_ms)
        if result.first_token_ms is not None:
            firsts.append(result.first_token_ms)

    payload = {
        "model": report.models[0],
        "backend": report.backend,
        "iterations": len(totals),
        "total_ms_mean": round(sum(totals) / len(totals), 3),
        "time_to_first_token_ms_mean": round(sum(firsts) / len(firsts), 3) if firsts else None,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
