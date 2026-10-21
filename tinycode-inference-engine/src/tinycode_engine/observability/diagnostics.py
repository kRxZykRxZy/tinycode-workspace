"""Diagnostics of the inference engine."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from ..backends.detection import discover_devices
from ..utils.platform import detect_architecture, detect_simd

__all__ = ["Diagnostics", "collect_diagnostics"]


@dataclass
class Diagnostics:
    """A snapshot of the runtime environment."""

    python_version: str = ""
    platform: str = ""
    architecture: str = ""
    simd: list[str] = field(default_factory=list)
    packages: dict[str, str] = field(default_factory=dict)
    missing_packages: list[str] = field(default_factory=list)
    devices: dict[str, Any] = field(default_factory=dict)
    native: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "architecture": self.architecture,
            "simd": list(self.simd),
            "packages": dict(self.packages),
            "missing_packages": list(self.missing_packages),
            "devices": dict(self.devices),
            "native": dict(self.native),
        }

    def to_text(self) -> str:
        lines = [
            "TinyCode inference engine diagnostics",
            "-------------------------------------",
            f"Python      : {self.python_version}",
            f"Platform    : {self.platform}",
            f"Architecture: {self.architecture}",
            f"SIMD        : {', '.join(self.simd) or 'none detected'}",
            "Packages    :",
        ]
        for name, version in sorted(self.packages.items()):
            lines.append(f"  - {name}: {version}")
        if self.missing_packages:
            lines.append("Optional packages that are not installed:")
            lines.extend(f"  - {name}" for name in sorted(self.missing_packages))
        lines.append("Devices     :")
        for key, value in sorted(self.devices.items()):
            lines.append(f"  - {key}: {value}")
        lines.append(f"Native      : {self.native}")
        return "\n".join(lines)


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover
        return None


def collect_diagnostics() -> Diagnostics:
    """Collect the diagnostics of the runtime environment."""

    from ..runtime.kernel_dispatch import KernelDispatcher

    diagnostics = Diagnostics(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        architecture=detect_architecture(),
        simd=detect_simd(),
    )
    for package in ("numpy", "tokenizers", "safetensors", "fastapi", "uvicorn", "torch", "openvino"):
        version = _package_version(package)
        if version:
            diagnostics.packages[package] = version
        else:
            diagnostics.missing_packages.append(package)

    devices = discover_devices().to_dict()
    diagnostics.devices = {
        "cuda": devices.get("cuda", {}).get("available", False),
        "cuda_reason": devices.get("cuda", {}).get("reason", ""),
        "graphics": [device.get("name") for device in devices.get("graphics", [])],
        "openvino": devices.get("openvino", {}).get("reason", ""),
        "neon": devices.get("neon", {}).get("available", False),
        "simd": devices.get("simd", []),
    }
    diagnostics.native = KernelDispatcher().describe()
    return diagnostics
