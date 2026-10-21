"""Integration test: export an artifact and verify it."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinycode_builder.export.pipeline import export_artifact
from tinycode_builder.export.verification import verify_artifact
from tinycode_builder.model.architecture import TransformerConfig
from tinycode_builder.model.causal_model import build_model


def test_export_and_verify(tmp_path: Path) -> None:
    from tinycode_builder.tokenizer.trainer import train_tokenizer_from_texts

    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer_from_texts(
        ["def f():\n    return 1\n" * 20, "class A:\n    pass\n" * 20],
        tokenizer_dir,
        vocab_size=300,
    )

    config = TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        context_length=1024,
        norm_type="rmsnorm",
    )
    model = build_model(config)
    target = model.num_parameters()

    output = tmp_path / "artifact"
    manifest = export_artifact(
        model,
        output,
        tokenizer_dir=tokenizer_dir,
        model_name="tinycode-test",
        parameter_target=target,
    )

    assert manifest["parameters"]["requested_target"] == target
    assert manifest["parameters"]["actual_trainable"] == target
    assert manifest["parameters"]["verified"] is True
    assert manifest["architecture"]["context_length"] == 1024

    report = verify_artifact(output)
    assert report.ok, report.summary()


def test_an_export_without_a_tokenizer_is_refused(tmp_path: Path) -> None:
    from tinycode_builder.utils.errors import ExportError

    config = TransformerConfig(
        vocab_size=256, d_model=32, n_layers=1, n_heads=4, d_ff=64, context_length=1024
    )
    model = build_model(config)
    with pytest.raises(ExportError):
        export_artifact(model, tmp_path / "artifact")


def test_export_includes_the_tokenizer(tmp_path: Path) -> None:
    from tinycode_builder.tokenizer.trainer import train_tokenizer_from_texts

    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer_from_texts(["def f():\n    return 1\n" * 20], tokenizer_dir, vocab_size=300)

    config = TransformerConfig(
        vocab_size=256, d_model=32, n_layers=1, n_heads=4, d_ff=64, context_length=1024
    )
    model = build_model(config)
    output = tmp_path / "artifact"
    manifest = export_artifact(model, output, tokenizer_dir=tokenizer_dir)

    assert manifest["tokenizer"]["fingerprint"].startswith("sha256:")
    assert (output / "tokenizer.json").is_file()
    assert verify_artifact(output).ok
