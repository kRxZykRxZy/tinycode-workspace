#!/usr/bin/env python
"""Benchmark a trained tokenizer.

Usage:
    python scripts/benchmark_tokenizer.py --tokenizer tokenizer/current --corpus data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_builder.tokenizer.benchmark import benchmark_files  # noqa: E402
from tinycode_builder.tokenizer.loader import load_tokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the TinyCode tokenizer.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--max-files", type=int, default=200)
    args = parser.parse_args()

    path = Path(args.tokenizer)
    path = path / "tokenizer.json" if path.is_dir() else path
    tokenizer = load_tokenizer(path)

    if args.corpus:
        corpus = Path(args.corpus)
        files = [str(item) for item in sorted(corpus.rglob("*")) if item.is_file()]
        result = benchmark_files(tokenizer, files, max_files=args.max_files)
    else:
        samples = [
            "def add(a, b):\n    return a + b\n",
            "class Example:\n    pass\n",
            "SELECT * FROM users WHERE id = ?;\n",
        ]
        from tinycode_builder.tokenizer.benchmark import benchmark_tokenizer

        result = benchmark_tokenizer(tokenizer, samples)

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
