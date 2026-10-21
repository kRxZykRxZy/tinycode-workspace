"""Artifact export pipeline.

Produces a directory that the inference engine can load:

    artifact.json          the versioned manifest
    model.safetensors      the weights (preferred)
    model.pt               the weights in PyTorch format (alternative)
    tokenizer.json         the tokenizer
    tokenizer_config.json  tokenizer metadata
    controller.*           optional trainable budget controller
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..utils.atomic_io import atomic_write_json
from ..utils.errors import ExportError
from .artifact_manifest import REQUIRED_OPERATORS, build_manifest
from .gguf_export import export_gguf
from .openvino_export import export_openvino_ir
from .pytorch_export import export_pytorch
from .safetensors_export import export_safetensors

__all__ = ["export_artifact"]


def export_artifact(
    model: Any,
    output_dir: str | Path,
    tokenizer_dir: str | Path | None = None,
    model_name: str = "tinycode",
    model_version: str = "0.1.0",
    parameter_target: int | None = None,
    controller_dir: str | Path | None = None,
    formats: tuple[str, ...] = ("safetensors", "pytorch"),
    training_metadata: dict[str, Any] | None = None,
    extra_manifest: dict[str, Any] | None = None,
    openvino: bool = False,
    openvino_device: str = "GPU",
) -> dict[str, Any]:
    """Export a complete, loadable artifact.

    Returns:
        The manifest dictionary that was written to ``artifact.json``.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    architecture = model.config.to_dict()
    actual_parameters = int(model.num_parameters())
    target = int(parameter_target if parameter_target is not None else actual_parameters)

    weights_entry: dict[str, Any] = {}
    files: dict[str, Path] = {}

    if "safetensors" in formats:
        entry = export_safetensors(model, output)
        weights_entry = entry
        files[entry["file"]] = Path(entry["path"])
    if "pytorch" in formats:
        entry = export_pytorch(model, output)
        files[entry["file"]] = Path(entry["path"])
        if not weights_entry:
            weights_entry = entry
    if not weights_entry:
        raise ExportError("No weight format was exported.")

    tokenizer_info: dict[str, Any] = {}
    if tokenizer_dir is None:
        raise ExportError(
            "A tokenizer directory is required: an artifact without a tokenizer cannot "
            "be served by the inference engine."
        )
    tokenizer_source = Path(tokenizer_dir)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = tokenizer_source / name
        if not source.is_file():
            continue
        destination = output / name
        shutil.copy2(source, destination)
        files[name] = destination
    from ..tokenizer.fingerprint import fingerprint_file

    tokenizer_path = output / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise ExportError(
            f"The tokenizer directory {tokenizer_source} does not contain a tokenizer.json."
        )
    tokenizer_info = {
        "fingerprint": fingerprint_file(tokenizer_path),
        "vocab_size": int(architecture.get("vocab_size", 0)),
        "files": {
            name: {"sha256": fingerprint_file(output / name)}
            for name in ("tokenizer.json", "tokenizer_config.json")
            if (output / name).is_file()
        },
    }

    controller_info: dict[str, Any] = {"present": False}
    if controller_dir is not None:
        controller_source = Path(controller_dir)
        if (controller_source / "controller.pt").is_file():
            for name in ("controller.pt", "controller.json"):
                source = controller_source / name
                if source.is_file():
                    destination = output / name
                    shutil.copy2(source, destination)
                    files[name] = destination
            from ..checkpoint.integrity import file_digest

            controller_info = {
                "present": True,
                "version": "1.0.0",
                "file": "controller.pt",
                "metadata_file": "controller.json",
                "feature_schema": "budget_features.v1",
                "sha256": file_digest(output / "controller.pt"),
            }

    from ..model.parameter_accounting import parameter_breakdown

    breakdown = parameter_breakdown(model.config).to_dict()

    openvino_info: dict[str, Any] = {"present": False}
    if openvino:
        openvino_dir = output / "openvino"
        entry = export_openvino_ir(model, openvino_dir, filename="model")
        for name in ("model.xml", "model.bin"):
            path = openvino_dir / name
            if path.is_file():
                files[f"openvino/{name}"] = path
        openvino_info = {
            "present": True,
            "format": "openvino-ir",
            "directory": "openvino",
            "file": "openvino/model.xml",
            "weights_file": "openvino/model.bin",
            "device": openvino_device,
            "opset": entry.get("opset"),
            "verified": bool(entry.get("verified")),
            "max_difference": entry.get("max_difference"),
            "note": (
                "The exported graph recomputes the whole sequence (no KV cache); it is a "
                "compatibility path for OpenVINO execution providers."
            ),
        }
        atomic_write_json(openvino_dir / "openvino_export.json", openvino_info)

    manifest = build_manifest(
        model_name=model_name,
        model_version=model_version,
        architecture=architecture,
        parameter_target=target,
        actual_parameters=actual_parameters,
        tokenizer_info=tokenizer_info,
        weights=weights_entry,
        files=files,
        controller=controller_info,
        training=training_metadata or {},
        export={
            "formats": list(formats),
            "tool": "tinycode-model-builder",
            "openvino": openvino_info,
        },
        compatibility={
            "min_engine_version": "0.1.0",
            "min_contract_version": "1.0.0",
            "supported_backends": ["cpu", "cuda", "intel_gpu", "arm_neon"],
            "required_operators": list(REQUIRED_OPERATORS),
        },
    )
    if extra_manifest:
        manifest.update(extra_manifest)

    atomic_write_json(output / "artifact.json", manifest)
    return manifest
