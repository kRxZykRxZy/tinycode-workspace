"""Integration test of the OpenVINO IR export."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinycode_builder.export.openvino_export import export_openvino_ir
from tinycode_builder.model.architecture import TransformerConfig
from tinycode_builder.model.causal_model import build_model
from tinycode_builder.utils.errors import ExportError

openvino = pytest.importorskip("openvino", reason="OpenVINO is not installed")


def _model() -> TransformerConfig:
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


def test_the_ir_is_exported_and_verified(tmp_path: Path) -> None:
    model = build_model(_model())
    entry = export_openvino_ir(model, tmp_path / "openvino", filename="model")

    assert (tmp_path / "openvino" / "model.xml").is_file()
    assert (tmp_path / "openvino" / "model.bin").is_file()
    assert entry["verified"] is True
    assert entry["max_difference"] is not None
    assert entry["max_difference"] < 1e-3


def test_the_export_is_part_of_the_artifact(tmp_path: Path) -> None:
    from tinycode_builder.export.pipeline import export_artifact
    from tinycode_builder.tokenizer.trainer import train_tokenizer_from_texts

    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer_from_texts(["def f():\n    return 1\n" * 20], tokenizer_dir, vocab_size=300)

    model = build_model(_model())
    output = tmp_path / "artifact"
    manifest = export_artifact(
        model,
        output,
        tokenizer_dir=tokenizer_dir,
        parameter_target=model.num_parameters(),
        openvino=True,
    )

    assert manifest["export"]["openvino"]["present"] is True
    assert manifest["export"]["openvino"]["verified"] is True
    assert (output / "openvino" / "model.xml").is_file()
    assert (output / "openvino" / "openvino_export.json").is_file()
