"""Verification of exported artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifact_manifest import validate_manifest

__all__ = ["VerificationReport", "verify_artifact", "compare_logits"]


@dataclass
class VerificationReport:
    """The outcome of verifying an artifact directory."""

    ok: bool = False
    directory: str = ""
    problems: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "directory": self.directory,
            "problems": list(self.problems),
            "checks": dict(self.checks),
        }

    def summary(self) -> str:
        status = "OK" if self.ok else "FAILED"
        lines = [f"Artifact verification: {status}", f"  directory: {self.directory}"]
        for name, value in self.checks.items():
            lines.append(f"  {name}: {value}")
        for problem in self.problems:
            lines.append(f"  problem: {problem}")
        return "\n".join(lines)


def verify_artifact(directory: str | Path, check_files: bool = True) -> VerificationReport:
    """Verify the structure, hashes and metadata of an exported artifact."""

    directory = Path(directory)
    report = VerificationReport(directory=str(directory))

    manifest_path = directory / "artifact.json"
    if not manifest_path.is_file():
        report.problems.append(f"manifest not found: {manifest_path}")
        return report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.problems.append(f"manifest is not valid JSON: {exc}")
        return report

    report.checks["artifact_format_version"] = manifest.get("artifact_format_version")
    problems = validate_manifest(manifest)
    report.problems.extend(problems)

    architecture = manifest.get("architecture", {}) or {}
    report.checks["context_length"] = architecture.get("context_length")
    report.checks["vocab_size"] = architecture.get("vocab_size")
    report.checks["parameter_target"] = (manifest.get("parameters", {}) or {}).get("requested_target")
    report.checks["actual_trainable"] = (manifest.get("parameters", {}) or {}).get("actual_trainable")

    if check_files:
        from ..checkpoint.integrity import file_digest

        files = manifest.get("files", {}) or {}
        for name, meta in files.items():
            path = directory / name
            if not path.is_file():
                report.problems.append(f"missing file: {name}")
                continue
            digest = file_digest(path)
            if digest != meta.get("sha256"):
                report.problems.append(f"hash mismatch for {name}")

        weights = manifest.get("weights", {}) or {}
        weights_file = weights.get("file")
        if weights_file:
            path = directory / weights_file
            if path.is_file():
                count = _count_parameters(path)
                report.checks["weight_tensor_count"] = count
                actual = (manifest.get("parameters", {}) or {}).get("actual_trainable")
                if isinstance(count, int) and isinstance(actual, int) and count != actual:
                    report.problems.append(
                        f"weight file contains {count} parameters but the manifest claims {actual}"
                    )

        tokenizer = manifest.get("tokenizer", {}) or {}
        tokenizer_files = tokenizer.get("files", {}) or {}
        for name, meta in tokenizer_files.items():
            path = directory / name
            if not path.is_file():
                report.problems.append(f"missing tokenizer file: {name}")
                continue
            digest = f"sha256:{file_digest(path)}"
            if digest != meta.get("sha256"):
                report.problems.append(f"tokenizer hash mismatch for {name}")

    report.ok = not report.problems
    return report


def _count_parameters(path: Path) -> int | None:
    try:
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            total = 0
            with safe_open(str(path), framework="pt") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    total += int(tensor.numel())
            return total
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(state, dict):
            return int(sum(int(v.numel()) for v in state.values() if hasattr(v, "numel")))
    except Exception:
        return None
    return None


def compare_logits(
    model_a: Any,
    model_b: Any,
    input_ids: Any,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Compare the logits of two models on the same input."""

    import torch

    with torch.no_grad():
        output_a = model_a(input_ids)
        output_b = model_b(input_ids)
    difference = (output_a.logits.float() - output_b.logits.float()).abs()
    max_difference = float(difference.max().item()) if difference.numel() else 0.0
    return {
        "max_absolute_difference": max_difference,
        "tolerance": tolerance,
        "within_tolerance": max_difference <= tolerance,
    }
