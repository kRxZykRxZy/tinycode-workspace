"""Primary entry point for the TinyCode Inference Engine.

Usage:

    python main.py serve          # start the OpenAI compatible server
    python main.py doctor         # print diagnostics
    python main.py bench          # run the local benchmark
    python main.py tokenize "..." # inspect the tokenizer

Without arguments the server is started on the local default endpoint
``http://127.0.0.1:8000/v1``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tinycode_engine.application.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
