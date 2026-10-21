"""Export a model from the builder and load it into the engine.

The test verifies the whole integration path: the builder builds a model,
exports the artifact and the engine loads it, validates the manifest and
reproduces the same logits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _build_config():
    from tinycode_builder.model.architecture import TransformerConfig

    return TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        context_length=1024,
        norm_type="rmsnorm",
        position_encoding="learned",
        qkv_bias=True,
        ff_bias=True,
        tie_embeddings=False,
    )


def _train_tokenizer(tmp_path: Path) -> Path:
    from tinycode_builder.tokenizer.trainer import train_tokenizer_from_texts

    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer_from_texts(
        ["def f():\n    return 1\n" * 20, "class A:\n    pass\n" * 20],
        tokenizer_dir,
        vocab_size=300,
    )
    return tokenizer_dir


def _build_and_export(tmp_path: Path, model: Any = None) -> Path:
    from tinycode_builder.export.pipeline import export_artifact
    from tinycode_builder.model.causal_model import build_model

    tokenizer_dir = _train_tokenizer(tmp_path)
    if model is None:
        model = build_model(_build_config())
    output = tmp_path / "artifact"
    export_artifact(
        model,
        output,
        tokenizer_dir=tokenizer_dir,
        model_name="tinycode-integration",
        parameter_target=model.num_parameters(),
    )
    return output


def test_the_engine_loads_the_exported_artifact(tmp_path: Path) -> None:
    from tinycode_engine.artifacts.loader import load_artifact

    artifact_dir = _build_and_export(tmp_path)
    artifact = load_artifact(artifact_dir, verify_integrity=True)
    assert artifact.model is not None
    assert artifact.manifest.context_length == 1024
    assert artifact.manifest.actual_parameters > 0


def test_the_manifest_is_validated(tmp_path: Path) -> None:
    from tinycode_engine.utils.errors import ArtifactError

    artifact_dir = _build_and_export(tmp_path)
    manifest_path = artifact_dir / "artifact.json"
    payload = manifest_path.read_text(encoding="utf-8").replace('"context_length": 1024', '"context_length": 4096')
    manifest_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ArtifactError):
        from tinycode_engine.artifacts.loader import load_artifact

        load_artifact(artifact_dir)


def test_the_logits_agree_within_the_tolerance(tmp_path: Path) -> None:
    import torch

    from tinycode_builder.model.causal_model import build_model

    torch_model = build_model(_build_config())
    artifact_dir = _build_and_export(tmp_path, model=torch_model)

    from tinycode_engine.artifacts.loader import load_artifact

    artifact = load_artifact(artifact_dir)
    assert artifact.model is not None

    input_ids = np.arange(16, dtype=np.int64)
    with torch.no_grad():
        torch_logits = torch_model(torch.as_tensor(input_ids)[None]).logits[0, -1].numpy()
    numpy_logits = artifact.model.forward(input_ids)

    difference = float(np.max(np.abs(torch_logits - numpy_logits)))
    assert difference < 1e-3, f"the logits differ by {difference}"


def test_the_golden_fixture_loads_in_the_engine(golden_artifact: Path) -> None:
    from tinycode_engine.artifacts.loader import load_artifact

    artifact = load_artifact(golden_artifact)
    assert artifact.manifest.context_length == 1024
    assert artifact.model is not None
    logits = artifact.model.forward(np.arange(8, dtype=np.int64))
    assert np.all(np.isfinite(logits))
