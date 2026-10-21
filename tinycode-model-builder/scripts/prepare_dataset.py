#!/usr/bin/env python
"""Prepare the packed dataset without using the interactive menu.

Usage:
    python scripts/prepare_dataset.py --local data/raw --output data/prepared
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_builder.configuration.loader import load_config  # noqa: E402
from tinycode_builder.data.pipeline import prepare_dataset  # noqa: E402
from tinycode_builder.tokenizer.loader import load_tokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the training dataset.")
    parser.add_argument("--config", default=None, help="YAML configuration file.")
    parser.add_argument("--local", default=None, help="Local corpus directory.")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--tokenizer", default=None, help="Directory with tokenizer.json.")
    parser.add_argument("--max-documents", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = None
    eos_token_id = 0
    if args.tokenizer:
        path = Path(args.tokenizer)
        path = path / "tokenizer.json" if path.is_dir() else path
        tokenizer = load_tokenizer(path)
        eos_token_id = tokenizer.token_to_id("<|endoftext|>") or 0

    result = prepare_dataset(
        config,
        args.output,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        source="local" if args.local else "the-stack",
        local_corpus=args.local,
        max_documents=args.max_documents or None,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
