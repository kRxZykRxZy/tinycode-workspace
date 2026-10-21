"""Hardware abstraction of the inference engine."""

from __future__ import annotations

from .capabilities import CapabilityReport, report_capabilities
from .discovery import DeviceInventory, discover_devices, discover_hardware
from .fallback import FallbackPlan, FallbackPolicy
from .selection import select, selection_report
from .telemetry import HardwareSample, HardwareTelemetry

__all__ = [
    "CapabilityReport",
    "DeviceInventory",
    "FallbackPlan",
    "FallbackPolicy",
    "HardwareSample",
    "HardwareTelemetry",
    "discover_devices",
    "discover_hardware",
    "report_capabilities",
    "select",
    "selection_report",
]
