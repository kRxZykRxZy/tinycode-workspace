"""Hardware discovery for the inference engine."""

from __future__ import annotations

from typing import Any

from ..backends.detection import DeviceInventory, discover_devices

__all__ = ["DeviceInventory", "discover_devices", "discover_hardware"]


def discover_hardware() -> dict[str, Any]:
    """Return a JSON friendly description of the hardware."""

    return discover_devices().to_dict()
