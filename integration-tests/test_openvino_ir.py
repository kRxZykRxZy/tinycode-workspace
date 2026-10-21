"""Cross repository test of the OpenVINO IR path.

The builder exports the IR, the engine loads and executes it.  The device
that is actually used is reported; on a machine whose OpenVINO GPU plugin
is not usable the test falls back to the CPU device and says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openvino", reason="OpenVINO is not installed")


def _config():
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


def test_the_exported_ir_runs_in_the_engine(tmp_path: Path) -> None:
    import numpy as np
    import torch

    from tinycode_builder.export.openvino_export import export_openvino_ir
    from tinycode_builder.model.causal_model import build_model
    from tinycode_engine.backends.intel_gpu.backend import IntelGpuBackend
    from tinycode_engine.backends.intel_gpu.provider import detect_openvino, import_openvino

    model = build_model(_config())
    entry = export_openvino_ir(model, tmp_path / "openvino", filename="model")
    assert entry["verified"] is True

    provider = detect_openvino()
    device = "GPU" if provider.gpu_plugin else "CPU"
    backend = IntelGpuBackend(
        model_path=tmp_path / "openvino" / "model.xml",
        device=device,
        allow_cpu_fallback=not provider.gpu_plugin,
    )
    backend.initialize()
    capabilities = backend.capabilities()
    assert capabilities.name == "intel_gpu"
    assert "device actually used" in capabilities.notes
    if not provider.gpu_plugin:
        # The machine cannot run the GPU plugin: the backend must not claim
        # GPU acceleration.
        assert capabilities.is_accelerated is False
        assert "does not claim GPU acceleration" in capabilities.notes

    ov = import_openvino()
    request = backend._compiled.create_infer_request()
    tokens = (np.arange(8, dtype=np.int64) % 256).reshape(1, 8)
    request.infer({"tokens": ov.Tensor(tokens)})
    logits = np.asarray(request.get_output_tensor(0).data).reshape(-1)

    with torch.no_grad():
        reference = model(torch.as_tensor(tokens, dtype=torch.long)).logits[0, -1].numpy()
    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-3, f"the IR output differs by {difference}"
    backend.shutdown()


def test_the_artifact_carries_the_ir(tmp_path: Path) -> None:
    from tinycode_builder.export.pipeline import export_artifact
    from tinycode_builder.model.causal_model import build_model
    from tinycode_builder.tokenizer.trainer import train_tokenizer_from_texts
    from tinycode_engine.artifacts.loader import load_artifact

    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer_from_texts(["def f():\n    return 1\n" * 20], tokenizer_dir, vocab_size=300)

    model = build_model(_config())
    output = tmp_path / "artifact"
    export_artifact(
        model,
        output,
        tokenizer_dir=tokenizer_dir,
        parameter_target=model.num_parameters(),
        openvino=True,
    )

    artifact = load_artifact(output)
    assert artifact.openvino.get("present") is True
    assert artifact.openvino_model_path is not None
    assert artifact.openvino_model_path.is_file()
