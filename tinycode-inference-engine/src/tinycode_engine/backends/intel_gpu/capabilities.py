"""Capabilities of the Intel integrated graphics backend."""

from __future__ import annotations

from typing import Any

from ..interface import BackendCapabilities
from .provider import detect_graphics_devices, detect_intel_graphics, detect_openvino

__all__ = ["intel_gpu_capabilities", "intel_gpu_status"]


def intel_gpu_status(verify_gpu_plugin: bool = False) -> dict[str, Any]:
    """Return the detection status of the Intel graphics backend.

    The GPU plugin is not loaded by default: a broken plugin can abort the
    process.  Pass ``verify_gpu_plugin=True`` to load it (only safe in a
    dedicated process).
    """

    graphics = detect_graphics_devices()
    intel = detect_intel_graphics()
    provider = detect_openvino(verify_gpu_plugin=verify_gpu_plugin)
    usable = bool(provider.available and provider.gpu_plugin)
    if usable:
        reason = "Intel graphics with a verified OpenVINO GPU plugin"
    elif not provider.available:
        reason = provider.reason or "OpenVINO is not available"
    elif not provider.gpu_plugin:
        reason = (
            "the OpenVINO GPU plugin is not available"
            + (f": {provider.gpu_plugin_error}" if provider.gpu_plugin_error else "")
        )
    else:
        reason = "Intel graphics execution requires Intel graphics and a verified GPU plugin"
    return {
        "graphics_devices": [device.to_dict() for device in graphics],
        "intel_graphics": [device.to_dict() for device in intel],
        "provider": provider.to_dict(),
        "usable": usable,
        "reason": reason,
    }


def intel_gpu_capabilities() -> BackendCapabilities:
    """Return the capabilities of the Intel GPU backend."""

    status = intel_gpu_status()
    return BackendCapabilities(
        name="intel_gpu",
        supported_operators=frozenset(),
        supported_precisions=frozenset(),
        supports_batching=False,
        supports_kv_cache=False,
        device_memory_bytes=0,
        is_accelerated=False,
        execution_provider="openvino",
        notes=status["reason"],
    )
