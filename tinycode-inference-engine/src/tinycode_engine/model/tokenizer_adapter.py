"""Adapter around the ``tokenizers`` library.

The adapter keeps the tokenisation behaviour identical to the model
builder: the same tokenizer file, the same special tokens and the same
chat prompt format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..utils.errors import ArtifactError

__all__ = ["TokenizerAdapter", "SPECIAL_TOKENS", "chat_prompt"]

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|code|>",
    "<|endofprompt|>",
)


def chat_prompt(
    messages: Sequence[dict[str, str]],
    add_generation_prompt: bool = True,
    system: str | None = None,
) -> str:
    """Render a chat history into the TinyCode prompt format."""

    parts: list[str] = []
    if system:
        parts.append(f"<|system|>\n{system}\n")
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "") or ""
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
        else:
            parts.append(f"<|user|>\n{content}\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


@dataclass
class TokenizerAdapter:
    """Wraps a trained tokenizer."""

    tokenizer: Any
    path: Path | None = None
    fingerprint: str = ""

    @classmethod
    def from_file(cls, path: str | Path, fingerprint: str = "") -> "TokenizerAdapter":
        """Load a ``tokenizer.json`` file."""

        try:
            from tokenizers import Tokenizer
        except Exception as exc:  # pragma: no cover - dependency missing
            raise ArtifactError(f"The 'tokenizers' package is required: {exc}") from exc

        target = Path(path)
        if target.is_dir():
            target = target / "tokenizer.json"
        if not target.is_file():
            raise ArtifactError(f"Tokenizer file not found: {target}")
        try:
            tokenizer = Tokenizer.from_file(str(target))
        except Exception as exc:
            raise ArtifactError(f"Failed to load tokenizer {target}: {exc}") from exc
        return cls(tokenizer=tokenizer, path=target, fingerprint=fingerprint)

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def token_id(self, token: str, default: int = 0) -> int:
        """Return the id of a special token (or ``default``)."""

        try:
            value = self.tokenizer.token_to_id(token)
        except Exception:  # pragma: no cover
            value = None
        return int(value) if value is not None else default

    @property
    def eos_token_id(self) -> int:
        return self.token_id("<|endoftext|>")

    @property
    def pad_token_id(self) -> int:
        return self.token_id("<|pad|>")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token ids."""

        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return list(encoding.ids)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Decode token ids back into text."""

        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def truncate(self, ids: Sequence[int], limit: int, keep_last: bool = True) -> list[int]:
        """Truncate a token sequence to ``limit`` tokens."""

        ids = list(ids)
        if len(ids) <= limit:
            return ids
        return ids[-limit:] if keep_last else ids[:limit]

    def describe(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "path": str(self.path) if self.path else None,
            "fingerprint": self.fingerprint,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
