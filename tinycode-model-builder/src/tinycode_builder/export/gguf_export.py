"""GGUF export.

A minimal, self contained GGUF v3 writer is implemented so that the
builder can produce a real GGUF file without depending on an external
converter.  The supported subset is deliberately narrow and verified:

* the architecture must use ``rmsnorm`` (GGUF has no LayerNorm bias),
* positional encodings must be learned,
* tensor data is written as F32 or F16.

Anything outside that subset raises a clear error instead of producing a
file that other tools would misread.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

__all__ = ["GGUF_VERSION", "GGML_TYPE_F32", "GGML_TYPE_F16", "export_gguf", "GGUFExportError"]

GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_BOOL = 1

_ALIGNMENT = 32


class GGUFExportError(RuntimeError):
    """Raised when the architecture cannot be represented in GGUF."""


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def _write_string(stream: Any, text: str) -> None:
    payload = text.encode("utf-8")
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)


def _write_value(stream: Any, value: Any, value_type: int) -> None:
    if value_type == _GGUF_TYPE_UINT32:
        stream.write(struct.pack("<I", int(value)))
    elif value_type == _GGUF_TYPE_INT32:
        stream.write(struct.pack("<i", int(value)))
    elif value_type == _GGUF_TYPE_FLOAT32:
        stream.write(struct.pack("<f", float(value)))
    elif value_type == _GGUF_TYPE_BOOL:
        stream.write(struct.pack("<B", 1 if value else 0))
    elif value_type == _GGUF_TYPE_STRING:
        _write_string(stream, str(value))
    elif value_type == _GGUF_TYPE_ARRAY:
        element_type, elements = value
        stream.write(struct.pack("<I", element_type))
        stream.write(struct.pack("<Q", len(elements)))
        for element in elements:
            _write_value(stream, element, element_type)
    else:  # pragma: no cover - defensive
        raise GGUFExportError(f"Unsupported GGUF value type: {value_type}")


def _kv(stream: Any, key: str, value: Any, value_type: int) -> None:
    _write_string(stream, key)
    stream.write(struct.pack("<I", value_type))
    _write_value(stream, value, value_type)


def _pad_to_alignment(stream: Any, alignment: int = _ALIGNMENT) -> None:
    position = stream.tell()
    remainder = position % alignment
    if remainder:
        stream.write(b"\x00" * (alignment - remainder))


# --------------------------------------------------------------------------
# tensor naming
# --------------------------------------------------------------------------
def map_tensor_name(name: str) -> str:
    """Map a TinyCode parameter name onto a GGUF tensor name."""

    if name == "embeddings.token.weight":
        return "token_emb.weight"
    if name == "embeddings.position.weight":
        return "position_emb.weight"
    if name == "final_norm.weight":
        return "output_norm.weight"
    if name in {"lm_head.lm_head.weight", "lm_head.weight"}:
        return "output.weight"
    if name.startswith("layers."):
        parts = name.split(".")
        index = parts[1]
        suffix = ".".join(parts[2:])
        mapping = {
            "norm1.weight": "attn_norm.weight",
            "attention.q_proj.weight": "attn_q.weight",
            "attention.k_proj.weight": "attn_k.weight",
            "attention.v_proj.weight": "attn_v.weight",
            "attention.o_proj.weight": "attn_output.weight",
            "norm2.weight": "ffn_norm.weight",
            "feed_forward.weight": "ffn_up.weight",
            "feed_forward_out.weight": "ffn_down.weight",
        }
        if suffix in mapping:
            return f"blk.{index}.{mapping[suffix]}"
    return name


def _f16_bytes(tensor: Any) -> bytes:
    import numpy as np

    array = tensor.detach().cpu().to(torch.float32).numpy() if hasattr(tensor, "detach") else tensor
    return np.asarray(array, dtype=np.float32).astype("<f2").tobytes()


def _f32_bytes(tensor: Any) -> bytes:
    import numpy as np

    array = tensor.detach().cpu().to(torch.float32).numpy() if hasattr(tensor, "detach") else tensor
    return np.asarray(array, dtype="<f4").tobytes()


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export_gguf(
    model: Any,
    output_dir: str | Path,
    filename: str = "model.gguf",
    dtype: str = "f32",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a model to a GGUF v3 file.

    Args:
        model: the TinyCode model (PyTorch module).
        output_dir: target directory.
        filename: name of the produced file.
        dtype: ``"f32"`` or ``"f16"``.
        metadata: additional metadata entries.

    Returns:
        A dictionary describing the exported file.
    """

    config = model.config
    if config.norm_type != "rmsnorm":
        raise GGUFExportError(
            "GGUF export currently supports rmsnorm only; LayerNorm has a bias that "
            "the GGUF Llama tensor layout cannot represent."
        )
    if config.position_encoding != "learned":
        raise GGUFExportError(
            "GGUF export currently supports learned positional embeddings only."
        )
    dtype_key = (dtype or "f32").lower()
    if dtype_key not in {"f32", "float32", "f16", "float16"}:
        raise GGUFExportError(f"Unsupported GGUF dtype: {dtype!r}")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename

    state = model.state_dict()
    tensor_entries: list[tuple[str, Any, tuple[int, ...]]] = []
    for name, tensor in state.items():
        if not hasattr(tensor, "shape"):
            continue
        if len(tensor.shape) != 2:
            continue
        tensor_entries.append((map_tensor_name(name), tensor, tuple(int(s) for s in tensor.shape)))

    kv_entries: list[tuple[str, Any, int]] = [
        ("general.architecture", "llama", _GGUF_TYPE_STRING),
        ("general.name", "tinycode", _GGUF_TYPE_STRING),
        ("general.file_type", 0 if dtype_key in {"f32", "float32"} else 1, _GGUF_TYPE_UINT32),
        ("llama.block_count", int(config.n_layers), _GGUF_TYPE_UINT32),
        ("llama.context_length", int(config.context_length), _GGUF_TYPE_UINT32),
        ("llama.embedding_length", int(config.d_model), _GGUF_TYPE_UINT32),
        ("llama.feed_forward_length", int(config.d_ff), _GGUF_TYPE_UINT32),
        ("llama.attention.head_count", int(config.n_heads), _GGUF_TYPE_UINT32),
        ("llama.attention.head_count_kv", int(config.n_heads), _GGUF_TYPE_UINT32),
        ("llama.attention.layer_norm_rms_epsilon", float(config.norm_eps), _GGUF_TYPE_FLOAT32),
        ("llama.vocab_size", int(config.vocab_size), _GGUF_TYPE_UINT32),
        ("general.quantization_version", 0, _GGUF_TYPE_UINT32),
    ]
    for key, value in (metadata or {}).items():
        kv_entries.append((key, value, _GGUF_TYPE_STRING))

    tensor_type = GGML_TYPE_F32 if dtype_key in {"f32", "float32"} else GGML_TYPE_F16

    with path.open("wb") as stream:
        stream.write(b"GGUF")
        stream.write(struct.pack("<I", GGUF_VERSION))
        stream.write(struct.pack("<Q", len(tensor_entries)))
        stream.write(struct.pack("<Q", len(kv_entries)))

        for key, value, value_type in kv_entries:
            _kv(stream, key, value, value_type)

        offset = 0
        tensor_info: list[tuple[str, tuple[int, ...], int, int]] = []
        for name, tensor, shape in tensor_entries:
            tensor_info.append((name, shape, tensor_type, offset))
            if tensor_type == GGML_TYPE_F32:
                size = tensor.numel() * 4
            else:
                size = tensor.numel() * 2
            offset += size
            padding = (-offset) % _ALIGNMENT
            offset += padding

        for name, shape, ggml_type, tensor_offset in tensor_info:
            _write_string(stream, name)
            stream.write(struct.pack("<I", len(shape)))
            for dim in shape:
                stream.write(struct.pack("<Q", dim))
            stream.write(struct.pack("<I", ggml_type))
            stream.write(struct.pack("<Q", tensor_offset))

        _pad_to_alignment(stream)

        for (name, tensor, shape), (_, _, ggml_type, _) in zip(tensor_entries, tensor_info):
            if ggml_type == GGML_TYPE_F32:
                stream.write(_f32_bytes(tensor))
            else:
                stream.write(_f16_bytes(tensor))
            _pad_to_alignment(stream)

    from ..checkpoint.integrity import file_digest

    return {
        "format": "gguf",
        "file": path.name,
        "path": str(path),
        "sha256": file_digest(path),
        "size": path.stat().st_size,
        "dtype": "float32" if tensor_type == GGML_TYPE_F32 else "float16",
        "tensors": len(tensor_entries),
        "experimental": True,
    }
