"""Tests for the tokenizer subsystem."""

from __future__ import annotations

from pathlib import Path

from tinycode_builder.tokenizer.fingerprint import fingerprint_file
from tinycode_builder.tokenizer.loader import decode_ids, encode_text, truncate_ids
from tinycode_builder.tokenizer.special_tokens import SPECIAL_TOKENS, chat_prompt
from tinycode_builder.tokenizer.trainer import train_tokenizer_from_texts


def test_train_encode_decode_round_trip(tmp_path: Path) -> None:
    texts = [
        "def add(a, b):\n    return a + b\n" * 8,
        "class Widget:\n    pass\n" * 8,
        "const x = () => 42;\n" * 8,
    ]
    result = train_tokenizer_from_texts(texts, tmp_path / "tok", vocab_size=600)
    assert result.vocab_size > 0
    assert result.fingerprint.startswith("sha256:")

    from tinycode_builder.tokenizer.loader import load_tokenizer

    tokenizer = load_tokenizer(tmp_path / "tok" / "tokenizer.json")
    ids = encode_text(tokenizer, "def add(a, b): return a + b")
    assert ids
    assert isinstance(decode_ids(tokenizer, ids), str)


def test_special_tokens_are_registered(tmp_path: Path) -> None:
    train_tokenizer_from_texts(["x = 1\n" * 20], tmp_path / "tok", vocab_size=300)
    from tinycode_builder.tokenizer.loader import load_tokenizer

    tokenizer = load_tokenizer(tmp_path / "tok" / "tokenizer.json")
    for token in SPECIAL_TOKENS:
        assert tokenizer.token_to_id(token) is not None


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")
    assert fingerprint_file(first) != fingerprint_file(second)


def test_truncate_keeps_the_requested_side() -> None:
    ids = list(range(10))
    assert truncate_ids(ids, 4, keep_last=True) == [6, 7, 8, 9]
    assert truncate_ids(ids, 4, keep_last=False) == [0, 1, 2, 3]
    assert truncate_ids(ids, 20) == ids


def test_chat_prompt_uses_the_contract_format() -> None:
    prompt = chat_prompt([{"role": "user", "content": "hello"}])
    assert prompt.startswith("<|user|>\n")
    assert prompt.endswith("<|assistant|>\n")


def test_chat_prompt_includes_system_messages() -> None:
    prompt = chat_prompt(
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    )
    assert "<|system|>\nbe terse\n" in prompt
    assert "<|user|>\nhi\n" in prompt
