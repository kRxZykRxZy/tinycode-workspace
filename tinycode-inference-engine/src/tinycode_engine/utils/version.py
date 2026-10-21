"""Version information and contract compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["engine_version", "ContractVersion", "parse_version", "version_tuple"]


def engine_version() -> str:
    """Return the version of the inference engine."""

    from .. import __version__

    return __version__


def parse_version(value: str) -> tuple[int, ...]:
    """Parse a semantic version string into a comparable tuple."""

    parts: list[int] = []
    for chunk in str(value).split("."):
        digits = ""
        for character in chunk:
            if character.isdigit():
                digits += character
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def version_tuple(value: str) -> tuple[int, int, int]:
    major, minor, patch = parse_version(value)
    return major, minor, patch


@dataclass(frozen=True)
class ContractVersion:
    """The artifact contract version supported by this engine."""

    version: str = "1.0.0"

    def supports(self, other: str) -> bool:
        """Return whether a manifest with ``other`` can be loaded."""

        return version_tuple(other)[:2] <= version_tuple(self.version)[:2]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.version
