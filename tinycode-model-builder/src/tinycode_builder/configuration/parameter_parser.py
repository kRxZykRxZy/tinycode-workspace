"""Parsing and validation of the user-selected parameter target.

The parameter target is a user choice and is therefore *not* hardcoded.
Supported shorthand values are parsed with correct handling of the ``K``,
``M`` and ``B`` suffixes as well as explicit integer counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "SUPPORTED_PARAMETER_TARGETS",
    "DEFAULT_PARAMETER_TARGET",
    "ParameterTargetError",
    "ParsedParameterTarget",
    "parse_parameter_target",
    "format_parameter_target",
    "is_supported_target",
]

#: Parameter targets the builder is able to construct architecture for.
SUPPORTED_PARAMETER_TARGETS: tuple[int, ...] = (
    1_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
)

#: Default target used when the user accepts the suggested value.
DEFAULT_PARAMETER_TARGET: int = 10_000_000

_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

_PATTERN = re.compile(r"^\s*([0-9][0-9_]*)\s*([kKmMbB])?\s*$")


class ParameterTargetError(ValueError):
    """Raised when a parameter target cannot be parsed or is unsupported."""


@dataclass(frozen=True)
class ParsedParameterTarget:
    """The result of parsing a parameter target string."""

    value: int
    original: str
    suffix: str | None = None
    supported: bool = False

    @property
    def millions(self) -> float:
        return self.value / 1_000_000

    def __int__(self) -> int:  # pragma: no cover - trivial
        return self.value


def parse_parameter_target(text: str | int) -> ParsedParameterTarget:
    """Parse a parameter target such as ``"10M"``, ``"500m"`` or ``"1000000"``.

    Args:
        text: user supplied target.

    Returns:
        A :class:`ParsedParameterTarget` describing the parsed value.

    Raises:
        ParameterTargetError: if the value is malformed or not positive.
    """

    if isinstance(text, bool):  # bool is an int subclass; reject explicitly
        raise ParameterTargetError("Parameter target must be a number, not a boolean.")
    if isinstance(text, int):
        value = text
        original = str(text)
        suffix = None
    else:
        original = str(text)
        match = _PATTERN.match(original)
        if not match:
            raise ParameterTargetError(
                f"Could not interpret {original!r} as a parameter count. "
                "Use forms such as 10M, 500m, 1B or an explicit integer."
            )
        digits = match.group(1).replace("_", "")
        suffix = match.group(2)
        base = int(digits)
        value = base * (_MULTIPLIERS[suffix.lower()] if suffix else 1)

    if value <= 0:
        raise ParameterTargetError("Parameter target must be a positive number of parameters.")

    supported = value in SUPPORTED_PARAMETER_TARGETS
    return ParsedParameterTarget(value=value, original=original, suffix=suffix, supported=supported)


def is_supported_target(value: int) -> bool:
    """Return whether ``value`` is one of the officially supported targets."""

    return value in SUPPORTED_PARAMETER_TARGETS


def format_parameter_target(value: int) -> str:
    """Render a parameter count using the shorthand notation."""

    if value >= 1_000_000_000 and value % 1_000_000_000 == 0:
        return f"{value // 1_000_000_000}B"
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)
