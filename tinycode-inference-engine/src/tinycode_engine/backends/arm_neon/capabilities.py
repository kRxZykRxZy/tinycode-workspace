"""ARM NEON capability detection."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Any

from ...utils.platform import detect_architecture, is_arm

__all__ = ["NeonCapabilities", "detect_neon"]


@dataclass
class NeonCapabilities:
    """Whether the CPU supports NEON and which variant."""

    available: bool = False
    architecture: str = "unknown"
    variant: str = ""
    reason: str = ""

    @property
    def vector_bits(self) -> int:
        return 128

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "architecture": self.architecture,
            "variant": self.variant,
            "vector_bits": self.vector_bits,
            "reason": self.reason,
        }


def _linux_flags() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("flags") or line.lower().startswith("features"):
                    return line.lower()
    except OSError:
        pass
    return ""


def detect_neon() -> NeonCapabilities:
    """Detect NEON support on ARM CPUs."""

    architecture = detect_architecture()
    capabilities = NeonCapabilities(architecture=architecture)

    if not is_arm():
        capabilities.reason = "the CPU is not an ARM CPU"
        return capabilities

    if architecture == "arm64":
        capabilities.available = True
        capabilities.variant = "aarch64"
        capabilities.reason = "NEON is mandatory on AArch64"
        return capabilities

    flags = _linux_flags()
    if "neon" in flags or "asimd" in flags:
        capabilities.available = True
        capabilities.variant = "armv7"
        capabilities.reason = "NEON reported by the kernel"
        return capabilities

    if sys.platform == "darwin":  # pragma: no cover - platform dependent
        capabilities.available = True
        capabilities.variant = "armv7"
        capabilities.reason = "Apple ARM CPUs support NEON"
        return capabilities

    capabilities.reason = f"NEON was not reported by the kernel ({platform.machine()})"
    return capabilities
