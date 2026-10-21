"""Platform detection helpers."""

from __future__ import annotations

import os
import platform
import sys

__all__ = [
    "detect_architecture",
    "detect_simd",
    "is_arm",
    "is_x86",
    "cpu_count",
    "platform_info",
]


def detect_architecture() -> str:
    """Return a normalised CPU architecture string."""

    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    if machine.startswith("arm"):
        return "arm32"
    return machine or "unknown"


def is_x86() -> bool:
    """Return whether the machine is an x86 CPU."""

    return detect_architecture() in {"x86_64", "i386", "i686"}


def is_arm() -> bool:
    """Return whether the machine is an ARM CPU."""

    return detect_architecture().startswith("arm")


def detect_simd() -> list[str]:
    """Return the SIMD instruction sets that this CPU reports."""

    features: list[str] = []
    try:
        import numpy as np

        # numpy exposes the compiled CPU features through its build config.
        config = getattr(np, "__config__", None)
        if config is not None and hasattr(config, "show"):
            info = config.show(mode="dicts")  # type: ignore[call-arg]
            cpu_features = (info.get("SIMD Extensions") or {}).get("found", []) or []
            features.extend(str(item) for item in cpu_features)
    except Exception:  # pragma: no cover - numpy always present
        pass

    if not features:
        machine = platform.machine().lower()
        if machine in {"amd64", "x86_64"}:
            features = ["SSE2"]
        elif machine.startswith("arm"):
            features = ["NEON"]

    # Normalise the names used by different numpy builds.
    normalised: list[str] = []
    for feature in features:
        upper = feature.upper()
        if "AVX512" in upper:
            normalised.append("AVX512")
        elif "AVX2" in upper:
            normalised.append("AVX2")
        elif "AVX" in upper:
            normalised.append("AVX")
        elif "SSE4" in upper:
            normalised.append("SSE4")
        elif "SSE2" in upper:
            normalised.append("SSE2")
        elif "NEON" in upper:
            normalised.append("NEON")
        elif "ASIMD" in upper:
            normalised.append("NEON")
    if not normalised:
        normalised = ["SSE2"] if is_x86() else []
    return sorted(set(normalised))


def cpu_count() -> int:
    """Return the number of usable CPUs."""

    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1


def platform_info() -> dict[str, object]:
    """Return a summary of the host platform."""

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "architecture": detect_architecture(),
        "python": sys.version.split()[0],
        "simd": detect_simd(),
        "cpu_count": cpu_count(),
    }
