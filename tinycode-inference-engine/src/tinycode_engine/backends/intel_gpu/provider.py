"""Detection of Intel integrated graphics and the OpenVINO provider.

Intel integrated graphics can only be used through a supported execution
provider.  The engine verifies that OpenVINO is installed, that the GPU
plugin can actually be loaded and that the device is usable before it
claims GPU acceleration.

Both OpenVINO layouts are supported:

* the modern layout (``openvino.Core()``, OpenVINO 2024 and newer),
* the legacy layout (``openvino.runtime``, OpenVINO 2023 and older).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GraphicsDevice",
    "ProviderStatus",
    "detect_graphics_devices",
    "detect_intel_graphics",
    "detect_openvino",
    "import_openvino",
    "openvino_available_devices",
    "probe_gpu_plugin",
]


@dataclass
class GraphicsDevice:
    """A graphics adapter reported by the operating system."""

    name: str
    vendor: str = ""
    source: str = ""

    @property
    def is_intel(self) -> bool:
        lowered = self.name.lower()
        return "intel" in lowered or "hd graphics" in lowered or "iris" in lowered

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "source": self.source,
            "is_intel": self.is_intel,
        }


def _run(command: list[str], timeout: float = 8.0) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        return result.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_graphics_devices() -> list[GraphicsDevice]:
    """Enumerate the graphics adapters of the machine."""

    devices: list[GraphicsDevice] = []
    if sys.platform.startswith("win"):
        output = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ]
        )
        for line in output.splitlines():
            name = line.strip()
            if name:
                devices.append(GraphicsDevice(name=name, vendor="unknown", source="win32"))
    elif sys.platform == "darwin":
        output = _run(["system_profiler", "SPDisplaysDataType"])
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chipset Model:"):
                name = stripped.split(":", 1)[1].strip()
                devices.append(GraphicsDevice(name=name, vendor="unknown", source="system_profiler"))
    else:
        output = _run(["lspci"])
        for line in output.splitlines():
            if "VGA compatible controller" in line or "3D controller" in line:
                parts = line.split(":", 2)
                name = parts[2].strip() if len(parts) > 2 else line.strip()
                devices.append(GraphicsDevice(name=name, vendor="unknown", source="lspci"))
    return devices


def detect_intel_graphics() -> list[GraphicsDevice]:
    """Return the Intel graphics adapters of the machine."""

    return [device for device in detect_graphics_devices() if device.is_intel]


def import_openvino() -> Any:
    """Import OpenVINO, supporting the modern and the legacy layout."""

    try:
        import openvino as ov

        if hasattr(ov, "Core"):
            return ov
    except Exception:
        pass
    try:
        import openvino.runtime as ov  # type: ignore[no-redef]

        return ov
    except Exception as exc:
        raise ImportError(f"OpenVINO is not installed: {exc}") from exc


@dataclass
class ProviderStatus:
    """The state of the OpenVINO execution provider."""

    available: bool = False
    version: str | None = None
    devices: list[str] = field(default_factory=list)
    gpu_plugin: bool = False
    gpu_plugin_error: str = ""
    cpu_device: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "version": self.version,
            "devices": list(self.devices),
            "gpu_plugin": self.gpu_plugin,
            "gpu_plugin_error": self.gpu_plugin_error,
            "cpu_device": self.cpu_device,
            "reason": self.reason,
        }


def detect_openvino(verify_gpu_plugin: bool = True) -> ProviderStatus:
    """Probe for OpenVINO and its GPU plugin.

    The probe runs in a **separate process**: on a machine whose GPU
    plugin DLL is broken, loading it aborts the process with a Windows
    fatal exception.  The caller therefore always gets a report instead
    of a dead process.

    Args:
        verify_gpu_plugin: also verify the GPU plugin (the default).  The
            verification happens out of process, so it is safe.
    """

    status = ProviderStatus()
    try:
        ov = import_openvino()
    except Exception as exc:
        status.reason = str(exc)
        return status

    status.available = True
    status.version = getattr(ov, "__version__", None)

    if not verify_gpu_plugin:
        status.reason = "OpenVINO detected (the GPU plugin was not probed)"
        return status

    result = probe_gpu_plugin()
    status.devices = [str(device) for device in result.get("devices", [])]
    for device in status.devices:
        if device.upper() == "CPU":
            status.cpu_device = device
        if device.upper().startswith("GPU"):
            status.gpu_plugin = True
    if result.get("version") and not status.version:
        status.version = result["version"]
    if result.get("ok"):
        status.gpu_plugin = True
        status.reason = "OpenVINO detected with a usable GPU plugin"
    else:
        status.gpu_plugin_error = str(result.get("error") or "the GPU plugin is not usable")
        if result.get("crashed"):
            status.gpu_plugin_error += (
                " (the plugin probe process crashed; the plugin is most likely broken)"
            )
        status.reason = (
            "OpenVINO detected without a usable GPU plugin: " + status.gpu_plugin_error
        )
    return status


def _probe_gpu_plugin(ov: Any) -> str:
    """Load the GPU plugin in this process.

    Only used by callers that accept the risk of aborting the process; the
    engine uses :func:`probe_gpu_plugin` (a subprocess) instead.

    Returns an empty string when the plugin works, the error otherwise.
    """

    try:
        core = ov.Core()
        devices = list(core.available_devices)
        if any(str(device).upper().startswith("GPU") for device in devices):
            return ""
        core.get_property("GPU", "SUPPORTED_DEVICES")
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:400]


_PROBE_SCRIPT = """
import json

result = {"ok": False, "devices": [], "error": "", "version": None, "crashed": False}

try:
    try:
        import openvino as ov
    except Exception:
        import openvino.runtime as ov
    result["version"] = getattr(ov, "__version__", None)
    core = ov.Core()
    devices = list(core.available_devices)
    result["devices"] = [str(device) for device in devices]
    if any(str(device).upper().startswith("GPU") for device in devices):
        result["ok"] = True
    else:
        try:
            core.get_property("GPU", "SUPPORTED_DEVICES")
            result["ok"] = True
        except Exception as exc:
            result["error"] = ("%s: %s" % (type(exc).__name__, exc)).replace("\\n", " ")[:400]
except Exception as exc:
    result["error"] = ("%s: %s" % (type(exc).__name__, exc)).replace("\\n", " ")[:400]

print(json.dumps(result))
"""


_PROBE_CACHE: dict[str, Any] = {"timestamp": 0.0, "result": None}
_PROBE_CACHE_TTL = 300.0


def probe_gpu_plugin(timeout: float = 60.0, use_cache: bool = True) -> dict[str, Any]:
    """Verify the OpenVINO GPU plugin in a **separate process**.

    Loading a broken GPU plugin can abort the process (a Windows fatal
    exception in the plugin loader), so the check always runs in a
    subprocess: a crash is reported as a failed probe instead of taking
    the server down.

    The result is cached for a few minutes so that the backend selection
    does not start a probe process for every request.

    Returns:
        ``{"ok": bool, "devices": [...], "error": str, "crashed": bool,
        "version": str|None}``.
    """

    import subprocess
    import sys
    import time

    if use_cache:
        cached = _PROBE_CACHE.get("result")
        if cached is not None and (time.time() - float(_PROBE_CACHE["timestamp"])) < _PROBE_CACHE_TTL:
            return dict(cached)

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _cache_probe(
            {
                "ok": False,
                "devices": [],
                "error": "the GPU plugin probe timed out",
                "crashed": False,
                "version": None,
            }
        )
    except OSError as exc:
        return _cache_probe(
            {
                "ok": False,
                "devices": [],
                "error": str(exc),
                "crashed": False,
                "version": None,
            }
        )

    for line in reversed((completed.stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            import json

            payload = json.loads(line)
            payload["crashed"] = bool(completed.returncode != 0)
            if payload.get("crashed") and not payload.get("error"):
                payload["error"] = (
                    f"the GPU plugin probe process exited with {completed.returncode}"
                )
            return _cache_probe(payload)
        except Exception:
            continue
    return _cache_probe(
        {
            "ok": False,
            "devices": [],
            "error": (completed.stderr or "the GPU plugin probe produced no result")[-400:],
            "crashed": completed.returncode != 0,
            "version": None,
        }
    )


def _cache_probe(payload: dict[str, Any]) -> dict[str, Any]:
    import time

    _PROBE_CACHE["timestamp"] = time.time()
    _PROBE_CACHE["result"] = dict(payload)
    return payload


def openvino_available_devices() -> list[str]:
    """Return the devices OpenVINO can use."""

    return detect_openvino().devices
