"""Human readable capability report for the training environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .detection import HardwareProfile, detect_hardware

__all__ = ["CapabilityReport", "build_capability_report", "format_capability_report"]


@dataclass
class CapabilityReport:
    """Summary of what this machine can do for training."""

    profile: HardwareProfile
    notes: list[str]

    @property
    def can_train_on_gpu(self) -> bool:
        return self.profile.has_gpu

    @property
    def estimated_gpu_memory_gb(self) -> float:
        return self.profile.cuda.total_vram_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "notes": self.notes,
            "can_train_on_gpu": self.can_train_on_gpu,
            "estimated_gpu_memory_gb": round(self.estimated_gpu_memory_gb, 3),
        }

    def to_text(self) -> str:
        return format_capability_report(self)


def build_capability_report(profile: HardwareProfile | None = None) -> CapabilityReport:
    """Build a capability report, probing the hardware when not supplied."""

    profile = profile or detect_hardware()
    notes: list[str] = []
    if profile.has_gpu:
        names = ", ".join(dev["name"] for dev in profile.cuda.devices)
        notes.append(f"CUDA available: {names}")
        notes.append(
            "Recommended precision: "
            f"{profile.recommended_precision} (capability "
            f"{profile.cuda.devices[0]['capability'] if profile.cuda.devices else 'unknown'})"
        )
    else:
        notes.append(f"No CUDA device detected ({profile.cuda.reason}).")
        notes.append("Training will run on CPU; expect very low throughput.")
    notes.append(
        f"Host RAM: {profile.memory.total_ram_gb:.1f} GiB total, "
        f"{profile.memory.available_ram_gb:.1f} GiB available"
    )
    notes.append(f"Disk free: {profile.memory.disk_free_gb:.1f} GiB")
    if profile.is_colab:
        notes.append("Google Colab runtime detected; Drive storage is available.")
    return CapabilityReport(profile=profile, notes=notes)


def format_capability_report(report: CapabilityReport) -> str:
    """Render a capability report as plain text."""

    profile = report.profile
    lines = [
        "Hardware capability report",
        "---------------------------",
        f"Platform        : {profile.platform}",
        f"Python          : {profile.python_version}",
        f"CPU count       : {profile.cpu_count}",
        f"CUDA available  : {profile.cuda.available} ({profile.cuda.reason})",
    ]
    for device in profile.cuda.devices:
        lines.append(
            f"  - {device['name']}: {device['total_memory'] / (1024**3):.1f} GiB, "
            f"capability {device['capability']}"
        )
    lines.append(
        f"Host RAM        : {profile.memory.total_ram_gb:.1f} GiB total / "
        f"{profile.memory.available_ram_gb:.1f} GiB free"
    )
    lines.append(f"Disk free       : {profile.memory.disk_free_gb:.1f} GiB")
    lines.append(f"Colab runtime   : {profile.is_colab}")
    lines.append("Notes:")
    lines.extend(f"  - {note}" for note in report.notes)
    return "\n".join(lines)
