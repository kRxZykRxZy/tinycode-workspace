#!/usr/bin/env python
"""Validate an exported TinyCode artifact.

Usage:
    python scripts/validate_artifact.py artifacts/tinycode-latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinycode_builder.export.verification import verify_artifact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a TinyCode artifact.")
    parser.add_argument("directory", help="Artifact directory containing artifact.json")
    parser.add_argument("--json", action="store_true", help="Print the raw report as JSON.")
    args = parser.parse_args()

    report = verify_artifact(args.directory)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
