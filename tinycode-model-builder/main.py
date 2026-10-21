"""Primary entry point for the TinyCode Model Builder.

Usage:

    python main.py

The builder is an interactive application.  Ordinary model-building tasks are
completed through guided prompts; command line flags are optional and only
exist for scripting and testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(SRC))

from tinycode_builder.application.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
