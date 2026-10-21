"""Tests for the dataset preparation primitives."""

from __future__ import annotations

import pytest

from tinycode_builder.data.deduplication import DedupConfig, Deduplicator
from tinycode_builder.data.filtering import DocumentFilter, FilterConfig
from tinycode_builder.data.packing import Packer, pack_documents
from tinycode_builder.data.splitting import SplitConfig, assign_split, split_documents
from tinycode_builder.data.token_accounting import TokenAccounting, TrainingTokenTargetError


def test_filter_drops_short_and_long_documents() -> None:
    document_filter = DocumentFilter(FilterConfig(min_chars=10, max_chars=100))
    assert document_filter.keep({"content": "short"}) is False
    assert document_filter.keep({"content": "x" * 200}) is False
    assert document_filter.keep({"content": "x" * 50}) is True
    assert document_filter.stats.too_short == 1
    assert document_filter.stats.too_long == 1


def test_filter_respects_the_language_list() -> None:
    document_filter = DocumentFilter(FilterConfig(min_chars=1, languages=("python",)))
    assert document_filter.keep({"content": "code", "language": "python"}) is True
    assert document_filter.keep({"content": "code", "language": "java"}) is False


def test_deduplicator_removes_exact_duplicates() -> None:
    deduplicator = Deduplicator(DedupConfig(near_duplicates=False))
    assert deduplicator.keep({"content": "same text here"}) is True
    assert deduplicator.keep({"content": "same text here"}) is False
    assert deduplicator.stats.exact_duplicates == 1


def test_split_is_deterministic_and_disjoint() -> None:
    documents = [{"content": f"doc-{index}"} for index in range(200)]
    config = SplitConfig(validation_fraction=0.2)
    first = split_documents(documents, config)
    second = split_documents(documents, config)

    assert len(first.validation) == len(second.validation)
    train_ids = {id(doc) for doc in first.train}
    validation_ids = {id(doc) for doc in first.validation}
    assert not (train_ids & validation_ids)
    assert assign_split(documents[0], 0, config) == assign_split(documents[0], 0, config)


def test_packer_emits_fixed_length_sequences() -> None:
    packer = Packer(sequence_length=1024, eos_token_id=1)
    sequences = list(packer.add_document(list(range(600))))
    sequences.extend(packer.add_document(list(range(600))))
    sequences.extend(packer.flush())
    assert sequences
    for sequence in sequences:
        assert sequence.length == 1024


def test_pack_documents_counts_tokens() -> None:
    sequences, stats = pack_documents([list(range(1000)), list(range(1000))], eos_token_id=1)
    assert stats["sequences"] == len(sequences)
    assert all(sequence.length == 1024 for sequence in sequences)


def test_token_accounting_separates_validation_tokens() -> None:
    accounting = TokenAccounting(target_training_tokens=1000)
    accounting.commit_training(600)
    accounting.commit_validation(500)
    assert accounting.training_tokens == 600
    assert accounting.validation_tokens == 500
    assert accounting.progress == 0.6
    assert not accounting.reached_target


def test_token_accounting_rejects_an_unreachable_target() -> None:
    accounting = TokenAccounting(target_training_tokens=1000)
    with pytest.raises(TrainingTokenTargetError):
        accounting.ensure_target_reachable(10)
