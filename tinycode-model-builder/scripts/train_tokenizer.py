#!/usr/bin/env python
"""Train the TinyCode tokenizer from a directory of source files.

Usage:
    python scripts/train_tokenizer.py --corpus data/raw --output tokenizer/current
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_builder.tokenizer.trainer import train_tokenizer  # noqa: E402

TEXT_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".sql",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the TinyCode tokenizer.")
    parser.add_argument("--corpus", required=True, help="Directory with the corpus.")
    parser.add_argument("--output", default="tokenizer/current")
    parser.add_argument("--vocab-size", type=int, default=98304)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=20000)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"Corpus directory not found: {corpus}")
        return 1

    files = [
        str(path)
        for path in sorted(corpus.rglob("*"))
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ][: args.max_files]
    if not files:
        print(f"No source files found under {corpus}")
        return 1

    print(f"Training on {len(files)} files ...")
    result = train_tokenizer(
        files,
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
