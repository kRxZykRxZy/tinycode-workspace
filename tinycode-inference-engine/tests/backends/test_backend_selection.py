"""Tests of the backend selection."""

from __future__ import annotations

import pytest

from tinycode_engine.backends.arm_neon.backend import ArmNeonBackend
from tinycode_engine.backends.arm_neon.capabilities import detect_neon
from tinycode_engine.backends.cpu.backend import CpuBackend
from tinycode_engine.backends.cuda.backend import CudaBackend
from tinycode_engine.backends.cuda.capabilities import detect_cuda_capabilities
from tinycode_engine.backends.interface import BackendUnavailableError
from tinycode_engine.backends.intel_gpu.backend import IntelGpuBackend
from tinycode_engine.backends.intel_gpu.provider import detect_openvino
from tinycode_engine.backends.registry import BackendRegistry
from tinycode_engine.utils.platform import detect_architecture


def test_the_cpu_backend_is_always_available() -> None:
    backend = CpuBackend()
    backend.initialize()
    capabilities = backend.capabilities()
    assert capabilities.name == "cpu"
    assert "linear" in capabilities.supported_operators
    assert "rmsnorm" in capabilities.supported_operators
    backend.shutdown()


def test_the_registry_falls_back_to_the_cpu_backend() -> None:
    selection = BackendRegistry({"preferred": "auto"}).select()
    assert selection.backend is not None
    assert selection.name in {"cuda", "intel_gpu", "arm_neon", "cpu"}
    assert selection.reason
    selection.backend.shutdown()


def test_the_selection_reports_every_attempt() -> None:
    selection = BackendRegistry({"preferred": "auto"}).select()
    assert selection.attempts
    for attempt in selection.attempts:
        assert "backend" in attempt
        assert "ok" in attempt
    selection.backend.shutdown()


def test_cuda_is_not_claimed_without_a_device() -> None:
    capabilities = detect_cuda_capabilities()
    if not capabilities.available:
        backend = CudaBackend()
        with pytest.raises(BackendUnavailableError):
            backend.initialize()


def test_intel_gpu_is_not_claimed_without_a_verified_provider() -> None:
    provider = detect_openvino()
    if not (provider.available and provider.gpu_plugin):
        backend = IntelGpuBackend()
        with pytest.raises(BackendUnavailableError):
            backend.initialize()


def test_neon_is_only_used_on_arm() -> None:
    capabilities = detect_neon()
    if detect_architecture().startswith("arm"):
        assert capabilities.available
    else:
        assert not capabilities.available
        backend = ArmNeonBackend()
        with pytest.raises(BackendUnavailableError):
            backend.initialize()


def test_the_execution_test_runs_on_the_cpu_backend() -> None:
    backend = CpuBackend()
    backend.initialize()
    import numpy as np

    tensor = backend.allocate_tensor((2, 2))
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (2, 2)
    backend.shutdown()
