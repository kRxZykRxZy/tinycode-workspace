"""Kernel dispatch.

The engine prefers the native C++ kernels that are built from ``native/``
when the shared library is available.  When it is not, the numpy
implementations are used: the runtime stays fully functional without a
compiler.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = ["NativeLibrary", "KernelDispatcher", "native_library_candidates"]

_LIBRARY_NAMES = {
    "linux": ["libtinycode_native.so"],
    "darwin": ["libtinycode_native.dylib"],
    "win32": ["tinycode_native.dll"],
}


def native_library_candidates() -> list[Path]:
    """Return the paths that are searched for the native library."""

    names = _LIBRARY_NAMES.get(sys.platform, ["libtinycode_native.so"])
    roots = [
        Path(__file__).resolve().parents[3] / "native" / "build",
        Path(__file__).resolve().parents[3] / "native" / "build" / "lib",
        Path(__file__).resolve().parent / "_native",
        Path.cwd() / "build",
    ]
    override = os.environ.get("TINYCODE_NATIVE_LIB")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    for root in roots:
        for name in names:
            candidates.append(root / name)
    return candidates


class NativeLibrary:
    """A thin ctypes wrapper around the native kernels."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.library = ctypes.CDLL(str(path))
        self._bind()

    def _bind(self) -> None:
        library = self.library
        library.tinycode_version.restype = ctypes.c_int
        library.tinycode_version.argtypes = []
        library.tinycode_matmul_f32.restype = None
        library.tinycode_matmul_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.tinycode_rmsnorm_f32.restype = None
        library.tinycode_rmsnorm_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
        ]
        library.tinycode_softmax_f32.restype = None
        library.tinycode_softmax_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        library.tinycode_silu_f32.restype = None
        library.tinycode_silu_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        library.tinycode_add_f32.restype = None
        library.tinycode_add_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

    @property
    def version(self) -> int:
        return int(self.library.tinycode_version())


class KernelDispatcher:
    """Chooses between the native kernels and the numpy fallbacks."""

    def __init__(self, prefer_native: bool = True) -> None:
        self.library: NativeLibrary | None = None
        self.prefer_native = prefer_native
        self.errors: list[str] = []
        if prefer_native:
            self._load()

    def _load(self) -> None:
        for candidate in native_library_candidates():
            if not candidate.is_file():
                continue
            try:
                self.library = NativeLibrary(candidate)
                return
            except Exception as exc:  # pragma: no cover - platform dependent
                self.errors.append(f"{candidate}: {exc}")
        self.library = None

    @property
    def native_available(self) -> bool:
        return self.library is not None

    def describe(self) -> dict[str, Any]:
        return {
            "native_available": self.native_available,
            "library": str(self.library.path) if self.library else None,
            "version": self.library.version if self.library else None,
            "errors": list(self.errors),
        }

    # ------------------------------------------------------------------
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication ``a @ b`` with ``a`` of shape (M, K)."""

        if self.library is None:
            return np.matmul(a, b)
        a32 = np.ascontiguousarray(a, dtype=np.float32)
        b32 = np.ascontiguousarray(b, dtype=np.float32)
        if a32.ndim != 2 or b32.ndim != 2:
            return np.matmul(a, b)
        m, k = a32.shape
        k2, n = b32.shape
        if k != k2:
            return np.matmul(a, b)
        out = np.zeros((m, n), dtype=np.float32)
        self.library.library.tinycode_matmul_f32(
            a32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m,
            n,
            k,
        )
        return out

    def rms_norm(self, x: Any, gain: Any, eps: float = 1e-5) -> Any:
        """RMS normalisation of a 1-D buffer."""

        if self.library is None:
            variance = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
            return (x * (1.0 / np.sqrt(variance + eps))) * gain
        values = np.ascontiguousarray(x, dtype=np.float32)
        gain32 = np.ascontiguousarray(gain, dtype=np.float32)
        if values.ndim != 1 or gain32.ndim != 1 or values.size != gain32.size:
            variance = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
            return (x * (1.0 / np.sqrt(variance + eps))) * gain
        self.library.library.tinycode_rmsnorm_f32(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gain32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(values.size),
            ctypes.c_float(eps),
        )
        return values

    def softmax(self, x: Any) -> Any:
        """In-place softmax of a 1-D buffer (returns a new array)."""

        if self.library is None:
            shifted = x - np.max(x, axis=-1, keepdims=True)
            exponent = np.exp(shifted)
            return exponent / np.sum(exponent, axis=-1, keepdims=True)
        values = np.ascontiguousarray(x, dtype=np.float32)
        if values.ndim != 1:
            shifted = x - np.max(x, axis=-1, keepdims=True)
            exponent = np.exp(shifted)
            return exponent / np.sum(exponent, axis=-1, keepdims=True)
        self.library.library.tinycode_softmax_f32(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(values.size)
        )
        return values

    def silu(self, x: Any) -> Any:
        """SiLU activation of a 1-D buffer."""

        if self.library is None:
            values = np.asarray(x, dtype=np.float32)
            return values / (1.0 + np.exp(-values))
        values = np.ascontiguousarray(x, dtype=np.float32)
        if values.ndim != 1:
            values = np.asarray(x, dtype=np.float32)
            return values / (1.0 + np.exp(-values))
        self.library.library.tinycode_silu_f32(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(values.size)
        )
        return values
