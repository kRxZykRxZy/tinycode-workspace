"""OpenVINO IR export.

The builder produces an OpenVINO IR (``.xml`` + ``.bin``) of the TinyCode
decoder so that the inference engine can execute it through a verified
OpenVINO execution provider (the Intel integrated graphics backend).

The exported graph is deliberately explicit: the attention is written as
``matmul -> + causal mask -> softmax -> matmul`` instead of relying on a
framework specific fused attention operator.  That keeps the graph
portable and verifiable.

The exported graph recomputes the whole sequence for every call (the
graph has no KV cache).  It is therefore a *compatibility* path, not the
fast path: the numpy runtime with the KV cache stays the default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..utils.errors import ExportError

__all__ = ["export_openvino_ir", "OnnxExportWrapper", "build_onnx_wrapper"]


class OnnxExportWrapper(nn.Module):
    """A wrapper that exports the decoder with an explicit attention graph."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model
        config = model.config
        if not isinstance(config, dict):
            config = config.to_dict()
        self.vocab_size = int(config["vocab_size"])
        self.d_model = int(config["d_model"])
        self.n_layers = int(config["n_layers"])
        self.n_heads = int(config["n_heads"])
        self.d_ff = int(config["d_ff"])
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / (self.head_dim**0.5)
        self.norm_eps = float(config.get("norm_eps", 1e-5))
        self.norm_type = str(config.get("norm_type", "rmsnorm"))
        self.position_encoding = str(config.get("position_encoding", "learned"))

    # ------------------------------------------------------------------
    def _norm(self, x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        if self.norm_type == "rmsnorm":
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(variance + self.norm_eps) * module.weight
        return torch.nn.functional.layer_norm(
            x, (x.shape[-1],), module.weight, getattr(module, "bias", None), self.norm_eps
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the logits of the last position: ``(batch, vocab)``."""

        hidden = self.model.embeddings.token(tokens)
        if self.position_encoding == "learned":
            positions = self.model.embeddings.position.weight[: tokens.shape[1]]
            hidden = hidden + positions.unsqueeze(0)
        else:
            from ..model.embeddings import sinusoidal_positions

            positions = sinusoidal_positions(tokens.shape[1], self.d_model)
            hidden = hidden + torch.as_tensor(positions, dtype=hidden.dtype).unsqueeze(0)

        seq = tokens.shape[1]
        mask = torch.triu(
            torch.full((seq, seq), float("-inf"), dtype=hidden.dtype), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        for layer in self.model.layers:
            normed = self._norm(hidden, layer.norm1)
            query = layer.attention.q_proj(normed)
            key = layer.attention.k_proj(normed)
            value = layer.attention.v_proj(normed)

            batch = query.shape[0]
            query = query.reshape(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            key = key.reshape(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            value = value.reshape(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale + mask
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, value)
            context = context.transpose(1, 2).reshape(batch, seq, self.d_model)

            hidden = hidden + layer.attention.o_proj(context)
            normed = self._norm(hidden, layer.norm2)
            inner = layer.feed_forward(normed)
            inner = inner / (1.0 + torch.exp(-inner))
            hidden = hidden + layer.feed_forward_out(inner)

        hidden = self._norm(hidden, self.model.final_norm)
        head = self.model.lm_head.lm_head
        if head is None:
            logits = hidden[:, -1, :] @ self.model.embeddings.token.weight.T
        else:
            logits = head(hidden[:, -1, :])
        return logits


def build_onnx_wrapper(model: Any) -> OnnxExportWrapper:
    """Return the export wrapper of a model."""

    return OnnxExportWrapper(model)


def export_openvino_ir(
    model: Any,
    output_dir: str | Path,
    filename: str = "model",
    opset: int = 17,
    verify: bool = True,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Export the model as an OpenVINO IR.

    Args:
        model: the TinyCode model.
        output_dir: the directory that receives the IR.
        filename: the base name of the IR files.
        opset: the ONNX opset used for the export.
        verify: compare the IR output with the PyTorch output.
        tolerance: the accepted maximum difference of the verification.

    Returns:
        A dictionary describing the exported files.
    """

    ov = _import_openvino()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    onnx_path = target / f"{filename}.onnx"
    xml_path = target / f"{filename}.xml"
    bin_path = target / f"{filename}.bin"

    wrapper = build_onnx_wrapper(model)
    wrapper.eval()

    example = torch.zeros((1, 8), dtype=torch.long)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example,
                str(onnx_path),
                opset_version=opset,
                input_names=["tokens"],
                output_names=["logits"],
                dynamic_axes={"tokens": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                do_constant_folding=True,
                dynamo=False,
            )
    except TypeError:
        # Older torch versions do not know the ``dynamo`` switch.
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example,
                str(onnx_path),
                opset_version=opset,
                input_names=["tokens"],
                output_names=["logits"],
                dynamic_axes={"tokens": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                do_constant_folding=True,
            )
    except Exception as exc:
        raise ExportError(f"The ONNX export failed: {exc}") from exc

    try:
        ov_model = ov.convert_model(str(onnx_path))
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)
    except Exception as exc:
        raise ExportError(f"The OpenVINO conversion failed: {exc}") from exc

    result: dict[str, Any] = {
        "format": "openvino-ir",
        "file": xml_path.name,
        "weights_file": bin_path.name,
        "path": str(xml_path),
        "onnx_path": str(onnx_path),
        "opset": opset,
        "verified": False,
    }

    if verify:
        difference, error = _verify_ir(ov, xml_path, wrapper, model)
        result["verified"] = difference is not None
        result["max_difference"] = difference
        if error:
            result["verification_error"] = error
        if difference is None:
            raise ExportError(
                "The exported OpenVINO IR could not be verified"
                + (f": {error}" if error else ".")
            )
        if difference > tolerance:
            raise ExportError(
                "The exported OpenVINO IR does not reproduce the PyTorch output "
                f"(difference: {difference})."
            )

    result["sha256"] = _file_digest(xml_path)
    result["size"] = xml_path.stat().st_size
    return result


def _import_openvino() -> Any:
    """Import OpenVINO, supporting both the modern and the legacy layout."""

    try:
        import openvino as ov

        if hasattr(ov, "Core"):
            return ov
    except Exception:
        pass
    try:
        import openvino.runtime as ov  # type: ignore[no-redef]

        return ov
    except Exception as exc:
        raise ExportError(
            f"OpenVINO is required for the IR export: {exc}. Install the 'openvino' package."
        ) from exc


def _verify_ir(
    ov: Any,
    xml_path: Path,
    wrapper: OnnxExportWrapper,
    model: Any,
) -> tuple[float | None, str]:
    """Compare the IR output with the PyTorch output.

    Returns:
        ``(difference, error)``; the difference is ``None`` when the
        verification could not be performed.
    """

    try:
        core = ov.Core()
        compiled = core.compile_model(str(xml_path), "CPU")
        config = model.config
        vocab = max(1, int(config["vocab_size"] if isinstance(config, dict) else config.vocab_size))
        token_ids = (np.arange(8, dtype=np.int64) % vocab).reshape(1, 8)
        tensor = ov.Tensor(np.asarray(token_ids, dtype=np.int64))
        request = compiled.create_infer_request()
        request.infer({"tokens": tensor})
        ir_logits = np.asarray(request.get_output_tensor(0).data)
        with torch.no_grad():
            torch_logits = wrapper(torch.as_tensor(token_ids, dtype=torch.long)).numpy()
        return float(np.max(np.abs(ir_logits - torch_logits))), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:400]


def _file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ir_metadata(directory: str | Path) -> dict[str, Any]:
    """Read the metadata that the builder stores next to an IR."""

    path = Path(directory) / "openvino_export.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
