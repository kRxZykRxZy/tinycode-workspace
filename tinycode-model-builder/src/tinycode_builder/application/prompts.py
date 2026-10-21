"""Interactive prompt helpers.

All prompts degrade gracefully when stdin is not a terminal: the default
value is used instead of blocking forever.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

__all__ = [
    "PromptError",
    "is_interactive",
    "ask_text",
    "ask_int",
    "ask_float",
    "ask_choice",
    "ask_yes_no",
    "ask_parameter_target",
    "ask_token_target",
    "confirm",
]


class PromptError(RuntimeError):
    """Raised when a prompt cannot be answered."""


def is_interactive() -> bool:
    """Return whether stdin looks like an interactive terminal."""

    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except Exception:  # pragma: no cover
        return False


def _read(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default or ""
    except KeyboardInterrupt:  # pragma: no cover
        raise
    if not answer and default is not None:
        return default
    return answer


def ask_text(prompt: str, default: str | None = None) -> str:
    """Ask for a free form string."""

    return _read(prompt, default)


def ask_int(prompt: str, default: int | None = None, minimum: int | None = None, maximum: int | None = None) -> int:
    """Ask for an integer within the given bounds."""

    while True:
        raw = _read(prompt, str(default) if default is not None else None)
        try:
            value = int(str(raw).replace("_", "").replace(",", ""))
        except (TypeError, ValueError):
            print(f"  Please enter a whole number.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"  Value must be at most {maximum}.")
            continue
        return value


def ask_float(prompt: str, default: float | None = None, minimum: float | None = None) -> float:
    """Ask for a floating point number."""

    while True:
        raw = _read(prompt, str(default) if default is not None else None)
        try:
            value = float(str(raw).replace(",", ""))
        except (TypeError, ValueError):
            print("  Please enter a number.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Value must be at least {minimum}.")
            continue
        return value


def ask_choice(prompt: str, choices: Sequence[str], default: str | None = None) -> str:
    """Ask the user to pick one of ``choices``."""

    listing = ", ".join(choices)
    while True:
        raw = _read(f"{prompt} ({listing})", default)
        if raw in choices:
            return raw
        print(f"  Please choose one of: {listing}")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask a yes/no question."""

    raw = _read(f"{prompt} (y/n)", "y" if default else "n").lower()
    return raw in {"y", "yes", "true", "1"}


def confirm(prompt: str, default: bool = False) -> bool:
    """Ask for confirmation; used before expensive operations."""

    return ask_yes_no(prompt, default=default)


def ask_parameter_target(default: int = 10_000_000, supported: Sequence[int] | None = None) -> int:
    """Ask for the target number of trainable parameters.

    The parameter count is a user choice; it is never silently changed.
    """

    from ..configuration.parameter_parser import (
        SUPPORTED_PARAMETER_TARGETS,
        format_parameter_target,
        parse_parameter_target,
    )

    supported = list(supported or SUPPORTED_PARAMETER_TARGETS)
    print("  Supported parameter targets:")
    for value in supported:
        marker = " (default)" if value == default else ""
        print(f"    - {format_parameter_target(value)}{marker}")

    while True:
        raw = _read("Target number of trainable parameters", format_parameter_target(default))
        try:
            parsed = parse_parameter_target(raw)
        except Exception as exc:
            print(f"  {exc}")
            continue
        if parsed.value in supported:
            return parsed.value
        print(f"  {parsed.value:,} is not a supported target. Choose one of the listed values.")


def ask_token_target(default: int = 200_000_000) -> int:
    """Ask for the training token target."""

    print(f"  Default training token target: {default:,} tokens")
    return ask_int("Training token target", default=default, minimum=1_000)
