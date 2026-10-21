#!/usr/bin/env python
"""Measure how the adaptive budget affects the answer.

The script compares the baseline policy with the policies that allocate
more computation and reports the measured token usage and latency.  It
does *not* claim that more computation improves the quality: quality is
only reported when a verification signal is available.

Usage:
    python benchmarks/budget_quality.py --model ../artifacts/tinycode-latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_engine.budget.budget_allocator import BudgetAllocator, ResourceState  # noqa: E402
from tinycode_engine.budget.complexity_estimator import ComplexityEstimator  # noqa: E402
from tinycode_engine.budget.controller import BaselineBudgetController  # noqa: E402
from tinycode_engine.budget.request_analyzer import RequestAnalyzer  # noqa: E402
from tinycode_engine.orchestration.request_pipeline import InferenceRequest  # noqa: E402
from tinycode_engine.runtime.engine import EngineConfig, InferenceEngine  # noqa: E402
from tinycode_engine.server.startup import start_engine  # noqa: E402

PROMPTS = [
    "Write a Python function that reverses a string.",
    "Explain the difference between a list and a tuple.",
    "Fix the bug in this function: def f(x): return x / 0",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the effect of the budget policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    config = EngineConfig(model_paths=[args.model], default_model_alias="bench")
    engine = InferenceEngine(config)
    report = start_engine(config, engine)
    if not report.models:
        print("no model could be loaded", file=sys.stderr)
        return 1

    analyzer = RequestAnalyzer()
    allocator = BudgetAllocator(controller=BaselineBudgetController())
    estimator = ComplexityEstimator()

    rows = []
    for prompt in PROMPTS:
        prompt_ids = engine.tokenize(prompt)
        analyzed = analyzer.analyze("bench", prompt, len(prompt_ids), args.max_tokens)
        estimate = estimator.estimate(analyzed)
        allocation = allocator.allocate(
            analyzed, estimate, uncertainty=0.0, resources=ResourceState()
        )
        request = InferenceRequest(
            request_id="bench",
            model_id=report.models[0],
            prompt=prompt,
            prompt_ids=list(prompt_ids),
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
        result = engine.generate(request)
        rows.append(
            {
                "prompt": prompt[:48],
                "complexity": round(estimate.score, 4),
                "policy": allocation.selected_policy,
                "allocated_tokens": allocation.allocation.generation_tokens,
                "consumed_tokens": result.completion_tokens,
                "latency_ms": round(result.elapsed_ms, 3),
                "stop_reason": result.budget.stop_reason if result.budget else None,
            }
        )

    print(json.dumps({"results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
