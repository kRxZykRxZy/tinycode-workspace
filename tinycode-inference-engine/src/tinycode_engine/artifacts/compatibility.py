"""Compatibility checks between an artifact and this engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils.errors import ArtifactError
from ..utils.version import version_tuple
from .manifest import SUPPORTED_ARCHITECTURES, ArtifactManifest

__all__ = ["CompatibilityReport", "check_compatibility", "FIXED_CONTEXT_LENGTH"]

#: The context length is fixed at 1024 tokens in every component.
FIXED_CONTEXT_LENGTH = 1024


@dataclass
class CompatibilityReport:
    """Whether the engine can serve an artifact."""

    ok: bool = True
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_problem(self, message: str) -> None:
        self.problems.append(message)
        self.ok = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "warnings": list(self.warnings),
        }


def check_compatibility(
    manifest: ArtifactManifest,
    backend_operators: set[str] | frozenset[str] | None = None,
    engine_version: str = "0.1.0",
) -> CompatibilityReport:
    """Check whether the engine can load and serve an artifact."""

    report = CompatibilityReport()

    version = manifest.format_version
    if not version:
        report.add_problem("The manifest does not declare an artifact_format_version.")
    else:
        major, minor, _patch = version_tuple(version)
        if (major, minor) > (1, 0):
            report.add_problem(
                f"artifact format {version} is newer than this engine understands."
            )

    if manifest.architecture_id not in SUPPORTED_ARCHITECTURES:
        report.add_problem(
            f"Unsupported architecture {manifest.architecture_id!r}; supported: "
            f"{list(SUPPORTED_ARCHITECTURES)}."
        )

    if manifest.context_length != FIXED_CONTEXT_LENGTH:
        report.add_problem(
            f"The artifact declares a context length of {manifest.context_length}; this "
            f"engine only supports {FIXED_CONTEXT_LENGTH} tokens."
        )

    if manifest.parameter_target != manifest.actual_parameters:
        report.add_warning(
            "The manifest reports a parameter target that differs from the actual "
            "trainable parameter count."
        )

    weights = manifest.weights
    if not weights.get("file"):
        report.add_problem("The manifest does not reference a weight file.")
    if not weights.get("sha256"):
        report.add_warning("The manifest does not contain a weight checksum.")

    if not manifest.tokenizer_fingerprint:
        report.add_problem("The manifest does not contain a tokenizer fingerprint.")

    if backend_operators is not None:
        missing = [op for op in manifest.required_operators if op not in backend_operators]
        if missing:
            report.add_problem(
                "The selected backend does not implement the required operators: "
                + ", ".join(missing)
            )

    compatibility = manifest.raw.get("compatibility") or {}
    minimum_engine = compatibility.get("min_engine_version")
    if minimum_engine:
        if version_tuple(engine_version) < version_tuple(str(minimum_engine)):
            report.add_problem(
                f"The artifact requires engine version {minimum_engine}; this engine is "
                f"{engine_version}."
            )

    return report
