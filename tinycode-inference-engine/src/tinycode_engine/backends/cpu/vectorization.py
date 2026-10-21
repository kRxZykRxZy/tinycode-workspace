"""SIMD detection for the CPU backend."""

from __future__ import annotations

from dataclasses import dataclass

from ...utils.platform import detect_simd, is_arm, is_x86

__all__ = ["VectorizationInfo", "detect_vectorization"]


@dataclass
class VectorizationInfo:
    """Which SIMD instruction sets are usable."""

    features: list[str]
    architecture: str
    preferred: str = "scalar"

    @property
    def has_avx2(self) -> bool:
        return "AVX2" in self.features

    @property
    def has_avx512(self) -> bool:
        return "AVX512" in self.features

    @property
    def has_neon(self) -> bool:
        return "NEON" in self.features

    def to_dict(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "architecture": self.architecture,
            "preferred": self.preferred,
        }


def detect_vectorization() -> VectorizationInfo:
    """Detect the SIMD capabilities of this CPU."""

    from ...utils.platform import detect_architecture

    features = detect_simd()
    architecture = detect_architecture()
    preferred = "scalar"
    if "AVX512" in features:
        preferred = "avx512"
    elif "AVX2" in features:
        preferred = "avx2"
    elif "AVX" in features:
        preferred = "avx"
    elif "SSE4" in features:
        preferred = "sse4"
    elif "SSE2" in features:
        preferred = "sse2"
    elif "NEON" in features:
        preferred = "neon"
    return VectorizationInfo(features=features, architecture=architecture, preferred=preferred)


def is_supported_architecture() -> bool:
    """Return whether the architecture is supported by the CPU backend."""

    return is_x86() or is_arm()
