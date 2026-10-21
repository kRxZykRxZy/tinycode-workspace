"""Loading a complete artifact.

An artifact consists of

    artifact.json      the versioned manifest
    model.safetensors  the weights
    tokenizer.json     the tokenizer
    controller.pt      the optional trainable budget controller
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..model.loader import load_weights
from ..model.model_adapter import TinyCodeModel
from ..model.tokenizer_adapter import TokenizerAdapter
from ..utils.errors import ArtifactError
from .compatibility import check_compatibility
from .integrity import verify_manifest_files
from .manifest import ArtifactManifest, load_manifest
from .tokenizer import load_artifact_tokenizer

__all__ = ["LoadedArtifact", "load_artifact"]


@dataclass
class LoadedArtifact:
    """A validated, loadable artifact."""

    directory: Path
    manifest: ArtifactManifest
    weights: dict[str, Any]
    weights_format: str
    tokenizer: TokenizerAdapter | None
    model: TinyCodeModel | None = None
    controller: dict[str, Any] | None = None
    openvino: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return self.manifest.model_name

    @property
    def parameter_count(self) -> int:
        if self.manifest.actual_parameters:
            return self.manifest.actual_parameters
        return int(sum(array.size for array in self.weights.values()))

    @property
    def openvino_model_path(self) -> Path | None:
        """The path of the OpenVINO IR of the artifact, when there is one."""

        if not self.openvino or not self.openvino.get("present"):
            return None
        relative = self.openvino.get("file")
        if not relative:
            return None
        path = self.directory / str(relative)
        return path if path.is_file() else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "model_id": self.model_id,
            "model_version": self.manifest.model_version,
            "architecture_id": self.manifest.architecture_id,
            "context_length": self.manifest.context_length,
            "vocab_size": self.manifest.vocab_size,
            "parameter_count": self.parameter_count,
            "weights_format": self.weights_format,
            "controller_present": bool(self.controller),
            "openvino_ir": bool(self.openvino.get("present")),
            "warnings": list(self.warnings),
        }


def load_artifact(
    directory: str | Path,
    verify_integrity: bool = True,
    build_model: bool = True,
    engine_version: str = "0.1.0",
) -> LoadedArtifact:
    """Load an artifact directory and validate it.

    Raises:
        ArtifactError: when the artifact is missing, malformed or
            incompatible with this engine.
    """

    directory = Path(directory)
    if not directory.is_dir():
        raise ArtifactError(f"The artifact directory does not exist: {directory}")

    manifest = load_manifest(directory)

    report = check_compatibility(manifest, engine_version=engine_version)
    if not report.ok:
        raise ArtifactError(
            "The artifact is not compatible with this engine:\n- "
            + "\n- ".join(report.problems)
        )

    if verify_integrity:
        integrity = verify_manifest_files(directory, manifest)
        if not integrity.ok:
            raise ArtifactError(
                "The artifact failed its integrity check: "
                f"missing={integrity.missing} mismatched={integrity.mismatched}"
            )

    weights, weights_format = load_weights(directory)

    tokenizer: TokenizerAdapter | None = None
    try:
        tokenizer = load_artifact_tokenizer(directory, manifest)
    except ArtifactError as exc:
        report.add_warning(f"The artifact has no usable tokenizer: {exc}")

    controller: dict[str, Any] | None = None
    controller_info = manifest.controller
    if controller_info.get("present"):
        controller_path = directory / str(controller_info.get("file", "controller.pt"))
        if controller_path.is_file():
            controller = _load_controller(controller_path, controller_info)
        else:
            report.add_warning(
                "The manifest references a controller artifact that is not present; the "
                "baseline budget policy is used."
            )

    controller: dict[str, Any] | None = None
    openvino_info = dict((manifest.raw.get("export") or {}).get("openvino") or {})
    if openvino_info.get("present"):
        relative = openvino_info.get("file")
        if not relative or not (directory / str(relative)).is_file():
            report.add_warning(
                "The manifest references an OpenVINO IR that is not present; the "
                "OpenVINO execution path is not available for this artifact."
            )
            openvino_info = {"present": False}

    model: TinyCodeModel | None = None
    if build_model:
        model = TinyCodeModel(manifest.architecture, weights)
        actual = int(sum(array.size for array in weights.values()))
        if manifest.actual_parameters and actual != manifest.actual_parameters:
            report.add_warning(
                f"The weight file contains {actual} parameters while the manifest "
                f"declares {manifest.actual_parameters}."
            )

    return LoadedArtifact(
        directory=directory,
        manifest=manifest,
        weights=weights,
        weights_format=weights_format,
        tokenizer=tokenizer,
        model=model,
        controller=controller,
        openvino=openvino_info,
        warnings=list(report.warnings),
    )


def _load_controller(path: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Load the optional trainable budget controller."""

    import json

    payload: dict[str, Any] = {"path": str(path), "present": True}
    metadata_path = path.parent / str(info.get("metadata_file", "controller.json"))
    if metadata_path.is_file():
        try:
            payload["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload["metadata"] = {}
    try:
        import torch

        payload["state_dict"] = torch.load(path, map_location="cpu", weights_only=False)
        payload["format"] = "torch"
    except Exception as exc:
        payload["error"] = str(exc)
        payload["format"] = "unavailable"
    return payload
