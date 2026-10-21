"""Shared pytest fixtures for the model builder test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tinycode_builder.configuration.defaults import default_config  # noqa: E402
from tinycode_builder.model.architecture import TransformerConfig  # noqa: E402


@pytest.fixture()
def config() -> dict:
    """Return a default builder configuration."""

    return default_config()


@pytest.fixture()
def tiny_config() -> TransformerConfig:
    """Return a very small architecture used by the fast tests."""

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


@pytest.fixture()
def sample_corpus(tmp_path: Path) -> Path:
    """Create a small directory of source files."""

    directory = tmp_path / "corpus"
    directory.mkdir()
    samples = [
        "def add(a, b):\n    return a + b\n" * 4,
        "class Greeter:\n    def __init__(self, name):\n        self.name = name\n" * 4,
        "function mul(a, b) { return a * b; }\n" * 4,
        "public static int main(String[] args) { System.out.println(1); }\n" * 4,
    ]
    for index, sample in enumerate(samples):
        (directory / f"file-{index}.py").write_text(sample, encoding="utf-8")
    return directory
