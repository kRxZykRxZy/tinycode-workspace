#!/usr/bin/env python
"""Inspect a dataset directory or The Stack and print statistics.

Usage:
    python scripts/inspect_dataset.py --local data/raw
    python scripts/inspect_dataset.py --stack --subset default
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_builder.data.access_validation import validate_dataset_access  # noqa: E402
from tinycode_builder.data.filtering import DocumentFilter, FilterConfig  # noqa: E402
from tinycode_builder.data.pipeline import iter_local_documents  # noqa: E402
from tinycode_builder.data.schema_detection import detect_schema  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a training corpus.")
    parser.add_argument("--local", help="Directory with the corpus.")
    parser.add_argument("--stack", action="store_true", help="Inspect The Stack instead.")
    parser.add_argument("--subset", default="default")
    parser.add_argument("--repo-id", default="bigcode/the-stack-smol")
    parser.add_argument("--max-documents", type=int, default=200)
    args = parser.parse_args()

    if args.stack or not args.local:
        report = validate_dataset_access(repo_id=args.repo_id, subset=args.subset)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 1

    documents = list(iter_local_documents(args.local, max_documents=args.max_documents))
    if not documents:
        print("No documents found.")
        return 1

    schema = detect_schema(documents)
    document_filter = DocumentFilter(FilterConfig())
    kept = 0
    for document in documents:
        if document_filter.keep(document):
            kept += 1

    lengths = sorted(len(document.get("content") or "") for document in documents)
    print(
        json.dumps(
            {
                "documents": len(documents),
                "kept": kept,
                "filter_stats": document_filter.stats.to_dict(),
                "schema": schema.to_dict(),
                "characters": {
                    "min": lengths[0] if lengths else 0,
                    "median": lengths[len(lengths) // 2] if lengths else 0,
                    "max": lengths[-1] if lengths else 0,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
