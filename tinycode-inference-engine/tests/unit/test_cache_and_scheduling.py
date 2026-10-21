"""Tests of the caches and the scheduler primitives."""

from __future__ import annotations

from tinycode_engine.cache.eviction import LruCache
from tinycode_engine.cache.prefix_cache import PrefixCache
from tinycode_engine.cache.token_cache import TokenCache
from tinycode_engine.scheduling.admission import AdmissionController
from tinycode_engine.scheduling.cancellation import CancellationRegistry
from tinycode_engine.scheduling.priorities import Priority
from tinycode_engine.scheduling.request_queue import RequestQueue


def test_the_lru_cache_evicts_the_oldest_entry() -> None:
    cache = LruCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert len(cache) == 2


def test_the_token_cache_key_includes_the_tokenizer_fingerprint() -> None:
    cache = TokenCache()
    cache.put("model", "1.0", "sha256:aaa", "hello", [1, 2, 3])
    assert cache.get("model", "1.0", "sha256:aaa", "hello") == [1, 2, 3]
    assert cache.get("model", "1.0", "sha256:bbb", "hello") is None


def test_the_prefix_cache_key_includes_the_model_identity() -> None:
    cache = PrefixCache()
    cache.put("model-a", "1.0", "sha256:aaa", list(range(32)))
    assert cache.get("model-a", "1.0", "sha256:aaa", list(range(32))) is not None
    assert cache.get("model-b", "1.0", "sha256:aaa", list(range(32))) is None


def test_the_prefix_cache_skips_short_prefixes() -> None:
    cache = PrefixCache()
    cache.put("model", "1.0", "sha256:aaa", [1, 2, 3])
    assert cache.get("model", "1.0", "sha256:aaa", [1, 2, 3]) is None


def test_the_queue_is_bounded() -> None:
    queue = RequestQueue(capacity=2)
    assert queue.put("a", None)
    assert queue.put("b", None)
    assert not queue.put("c", None)
    assert queue.rejected == 1


def test_the_queue_is_ordered_by_priority() -> None:
    queue = RequestQueue(capacity=8)
    queue.put("low", None, priority=Priority.BACKGROUND)
    queue.put("high", None, priority=Priority.INTERACTIVE)
    item = queue.get()
    assert item is not None
    assert item.request_id == "high"


def test_a_cancelled_request_is_skipped() -> None:
    queue = RequestQueue(capacity=8)
    queue.put("a", None)
    queue.put("b", None)
    assert queue.cancel("a")
    item = queue.get()
    assert item is not None
    assert item.request_id == "b"


def test_the_admission_controller_limits_concurrency() -> None:
    controller = AdmissionController(max_active_requests=1)
    assert controller.check().admitted
    controller.on_start()
    decision = controller.check()
    assert not decision.admitted
    assert decision.status_code == 503


def test_cancellation_notifies_the_callbacks() -> None:
    registry = CancellationRegistry()
    called = {"value": False}

    def _callback() -> None:
        called["value"] = True

    registry.register("id", _callback)
    registry.cancel("id")
    assert called["value"]
    assert registry.is_cancelled("id")
    registry.release("id")
    assert not registry.is_cancelled("id")
