#!/usr/bin/env python
"""Run the workspace level benchmarks.

Usage:
    python benchmarks/run_all.py --model ../tinycode-model-builder/artifacts/tinycode-latest
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = WORKSPACE_ROOT / "tinycode-inference-engine"


def _run(script: str, arguments: list[str]) -> dict:
    command = [sys.executable, str(ENGINE_ROOT / "benchmarks" / script), *arguments]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"script": script, "error": "timeout"}
    if completed.returncode != 0:
        return {"script": script, "error": completed.stderr.strip()[-800:]}
    try:
        return {"script": script, "result": json.loads(completed.stdout)}
    except json.JSONDecodeError:
        return {"script": script, "output": completed.stdout.strip()[-800:]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every benchmark of the workspace.")
    parser.add_argument("--model", required=True, help="Path of the artifact directory.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    report = {
        "throughput": _run(
            "throughput.py",
            ["--model", args.model, "--max-tokens", str(args.max_tokens), "--iterations", str(args.iterations)],
        ),
        "latency": _run(
            "latency.py",
            ["--model", args.model, "--max-tokens", str(args.max_tokens), "--iterations", str(args.iterations)],
        ),
        "backend_comparison": _run("backend_comparison.py", ["--model", args.model]),
        "budget_quality": _run("budget_quality.py", ["--model", args.model]),
        "memory_usage": _run("memory_usage.py", ["--model", args.model]),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
