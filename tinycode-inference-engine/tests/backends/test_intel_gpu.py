"""Tests of the Intel integrated graphics backend.

The tests verify what the machine actually provides.  When the OpenVINO
GPU plugin cannot be loaded the backend must refuse to claim GPU
acceleration, and the tests assert exactly that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tinycode_engine.backends.intel_gpu.backend import IntelGpuBackend
from tinycode_engine.backends.intel_gpu.capabilities import intel_gpu_status
from tinycode_engine.backends.intel_gpu.provider import (
    detect_graphics_devices,
    detect_intel_graphics,
    detect_openvino,
    import_openvino,
)
from tinycode_engine.backends.interface import BackendUnavailableError

openvino = pytest.importorskip("openvino", reason="OpenVINO is not installed")


def _write_tiny_ir(directory: Path) -> Path:
    """Write a tiny OpenVINO IR with the interface the backend expects."""

    ov = import_openvino()
    ops = getattr(ov, "opset16", None) or getattr(ov, "opset13")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.xml"

    tokens = ops.parameter([1, 4], ov.Type.i32, name="tokens")
    table = ops.constant(
        [[float(index)] * 4 for index in range(8)], ov.Type.f32
    )
    gathered = ops.gather(table, tokens, axis=0)
    model = ov.Model([gathered], [tokens], "tiny-probe")
    ov.save_model(model, str(path), compress_to_fp16=False)
    return path


def test_the_graphics_devices_are_detected() -> None:
    devices = detect_graphics_devices()
    assert isinstance(devices, list)
    for device in devices:
        assert device.name


def test_the_openvino_provider_reports_its_devices() -> None:
    status = detect_openvino()
    assert status.available
    assert status.devices
    assert "CPU" in {device.upper() for device in status.devices}
    assert status.version


def test_the_gpu_plugin_is_only_claimed_when_it_works() -> None:
    status = detect_openvino()
    if status.gpu_plugin:
        assert not status.gpu_plugin_error
    else:
        # The reason has to be recorded, otherwise the refusal is silent.
        assert status.gpu_plugin_error or status.reason


def test_the_gpu_plugin_probe_runs_in_a_separate_process() -> None:
    """A broken plugin must not be able to abort the test process."""

    from tinycode_engine.backends.intel_gpu.provider import probe_gpu_plugin

    result = probe_gpu_plugin()
    assert "ok" in result
    assert "error" in result
    assert "devices" in result
    status = detect_openvino()
    if status.gpu_plugin:
        assert result["ok"] is True
    else:
        assert result["ok"] is False
        assert result["error"], "a failing probe has to report a reason"


def test_the_status_explains_why_the_backend_is_unavailable() -> None:
    status = intel_gpu_status()
    assert "reason" in status
    assert isinstance(status["usable"], bool)
    assert "graphics_devices" in status
    assert "provider" in status


def test_the_backend_refuses_to_run_without_a_usable_gpu_plugin(tmp_path: Path) -> None:
    status = detect_openvino()
    if status.gpu_plugin:
        pytest.skip("the GPU plugin is usable on this machine")
    backend = IntelGpuBackend(model_path=_write_tiny_ir(tmp_path))
    with pytest.raises(BackendUnavailableError):
        backend.initialize()


def test_the_backend_reports_the_device_it_really_used(tmp_path: Path) -> None:
    """The IR path has to work and the device has to be reported truthfully."""

    path = _write_tiny_ir(tmp_path)
    backend = IntelGpuBackend(model_path=path, device="CPU", allow_cpu_fallback=True)
    backend.initialize()
    capabilities = backend.capabilities()
    assert capabilities.name == "intel_gpu"
    assert capabilities.execution_provider.startswith("openvino:")
    assert "device actually used" in capabilities.notes
    if not capabilities.is_accelerated:
        assert "does not claim GPU acceleration" in capabilities.notes

    import numpy as np

    logits = backend.execute_forward(np.arange(4, dtype=np.int64))
    assert logits is not None
    assert np.all(np.isfinite(np.asarray(logits)))
    backend.shutdown()


def test_the_backend_reports_the_graphics_adapters_of_the_machine() -> None:
    status = intel_gpu_status()
    names = [device["name"] for device in status["graphics_devices"]]
    intel = [device["name"] for device in status["intel_graphics"]]
    assert len(intel) <= len(names)
    for device in detect_intel_graphics():
        assert device.is_intel
