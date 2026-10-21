"""Environment diagnostics."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from ..hardware.detection import detect_hardware

__all__ = ["Diagnostics", "collect_diagnostics"]


@dataclass
class Diagnostics:
    """A snapshot of the environment the builder runs in."""

    python_version: str = ""
    platform: str = ""
    executable: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    missing_packages: list[str] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "executable": self.executable,
            "packages": dict(self.packages),
            "missing_packages": list(self.missing_packages),
            "hardware": dict(self.hardware),
            "paths": dict(self.paths),
        }

    def to_text(self) -> str:
        lines = [
            "Builder diagnostics",
            "------------------",
            f"Python     : {self.python_version}",
            f"Platform   : {self.platform}",
            f"Executable : {self.executable}",
            "Packages   :",
        ]
        for name, version in sorted(self.packages.items()):
            lines.append(f"  - {name}: {version}")
        if self.missing_packages:
            lines.append("Missing (optional):")
            lines.extend(f"  - {name}" for name in sorted(self.missing_packages))
        lines.append("Hardware   :")
        for key, value in sorted(self.hardware.items()):
            lines.append(f"  - {key}: {value}")
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
    """Collect the environment diagnostics."""

    from ..utils.paths import project_root

    diagnostics = Diagnostics(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        executable=sys.executable,
    )
    for package in ("torch", "numpy", "tokenizers", "safetensors", "datasets", "pyarrow", "fastapi", "uvicorn"):
        version = _package_version(package)
        if version:
            diagnostics.packages[package] = version
        else:
            diagnostics.missing_packages.append(package)

    profile = detect_hardware()
    diagnostics.hardware = {
        "cuda_available": profile.cuda.available,
        "cuda_reason": profile.cuda.reason,
        "cpu_count": profile.cpu_count,
        "total_ram_gb": round(profile.memory.total_ram_gb, 2),
        "available_ram_gb": round(profile.memory.available_ram_gb, 2),
        "is_colab": profile.is_colab,
    }
    root = project_root()
    diagnostics.paths = {
        "project_root": str(root),
        "runs": str(root / "runs"),
        "artifacts": str(root / "artifacts"),
        "data": str(root / "data"),
    }
    return diagnostics
