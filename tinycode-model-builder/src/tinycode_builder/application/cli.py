"""Command line interface of the model builder.

The primary usage is interactive::

    python main.py

Flags exist for scripting and testing only; ordinary model building tasks
are completed through the guided prompts.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .menu import BUILDER_MENU
from .prompts import is_interactive
from .session import BuilderSession
from .workflows import (
    workflow_configure,
    workflow_controller,
    workflow_data,
    workflow_diagnostics,
    workflow_dashboard,
    workflow_evaluate,
    workflow_export,
    workflow_hardware,
    workflow_tokenizer,
    workflow_train,
)

__all__ = ["main", "build_parser"]

_WORKFLOWS = {
    "hardware": workflow_hardware,
    "configure": workflow_configure,
    "data": workflow_data,
    "tokenizer": workflow_tokenizer,
    "train": workflow_train,
    "evaluate": workflow_evaluate,
    "export": workflow_export,
    "controller": workflow_controller,
    "dashboard": workflow_dashboard,
    "diagnostics": workflow_diagnostics,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""

    parser = argparse.ArgumentParser(
        prog="tinycode-builder",
        description="Build, train and export TinyCode coding models.",
    )
    parser.add_argument(
        "--workflow",
        choices=sorted(_WORKFLOWS),
        help="Run a single workflow instead of the interactive menu.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repository root that holds runs/, artifacts/ and data/.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the builder version and exit.",
    )
    return parser


def _run_interactive(session: BuilderSession) -> int:
    print("TinyCode model builder")
    print("Run 'python main.py' and follow the prompts to build a model.")
    print("The context length is permanently fixed at 1024 tokens.\n")

    while True:
        print()
        print(BUILDER_MENU.render())
        try:
            choice = input("Select an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        item = BUILDER_MENU.item_for(choice)
        if item is None:
            if choice.lower() in {"q", "quit", "exit"}:
                print("Bye.")
                return 0
            print("  Unknown option.")
            continue

        handler = _WORKFLOWS.get(item.key)
        if handler is None:  # pragma: no cover - menu and workflows stay in sync
            print("  This option is not available.")
            continue

        try:
            handler(session)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        except Exception as exc:
            print(f"  The workflow failed: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the model builder."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        from .. import __version__

        print(f"tinycode-model-builder {__version__}")
        return 0

    session = BuilderSession(root=args.root).load()

    if args.workflow:
        handler = _WORKFLOWS[args.workflow]
        handler(session)
        return 0

    if not is_interactive():
        print("No interactive terminal detected.")
        print("Available workflows: " + ", ".join(sorted(_WORKFLOWS)))
        print("Run one with: python main.py --workflow <name>")
        return 2

    return _run_interactive(session)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
