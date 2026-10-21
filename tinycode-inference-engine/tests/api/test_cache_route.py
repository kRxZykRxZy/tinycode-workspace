"""Tests of the cache input route."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_caching_an_input_requires_authentication(client: TestClient) -> None:
    response = client.post("/v1/cache/input", json={"input": "hello"})
    assert response.status_code == 401


def test_an_input_can_be_cached(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/cache/input", headers=auth_headers, json={"input": "def add(a, b):"}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["cached"] is True
    assert payload["tokens"] > 0
    assert payload["prefix_state_stored"] is True
    assert payload["object"] == "cache.entry"


def test_an_empty_input_is_rejected(client: TestClient, auth_headers: dict) -> None:
    response = client.post("/v1/cache/input", headers=auth_headers, json={"input": "   "})
    assert response.status_code == 400


def test_a_request_without_an_input_is_rejected(client: TestClient, auth_headers: dict) -> None:
    response = client.post("/v1/cache/input", headers=auth_headers, json={})
    assert response.status_code == 400


def test_a_chat_conversation_can_be_cached(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/cache/input",
        headers=auth_headers,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["tokens"] > 0


def test_the_cache_statistics_are_exposed(client: TestClient, auth_headers: dict) -> None:
    client.post("/v1/cache/input", headers=auth_headers, json={"input": "hello world"})
    response = client.get("/v1/cache", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "cache.stats"
    assert "token_cache" in payload
    assert "prefix_cache" in payload


def test_the_cache_can_be_cleared(client: TestClient, auth_headers: dict) -> None:
    client.post("/v1/cache/input", headers=auth_headers, json={"input": "hello world"})
    response = client.delete("/v1/cache", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["cleared"] is True
    assert response.json()["prefix_cache"]["entries"] == 0


LONG_PROMPT = (
    "def add(a, b): return a + b "
    "def sub(a, b): return a - b "
    "def mul(a, b): return a * b "
    "def div(a, b): return a / b "
    "def mod(a, b): return a % b "
    "def pow_(a, b): return a ** b "
    "def neg(a): return -a "
    "def abs_(a): return abs(a) "
    "def min_(a, b): return min(a, b) "
    "def max_(a, b): return max(a, b) "
    "def sum_(a, b): return a + b "
    "def call(f, x): return f(x)"
)


def test_a_cached_prefix_is_reused(client: TestClient, auth_headers: dict, engine) -> None:
    prompt = LONG_PROMPT

    cached = client.post("/v1/cache/input", headers=auth_headers, json={"input": prompt})
    assert cached.status_code == 200, cached.text
    assert cached.json()["prefix_state_stored"] is True
    before = engine.prefix_cache.hits

    response = client.post(
        "/v1/completions",
        headers=auth_headers,
        json={
            "model": "tinycode-test",
            "prompt": prompt,
            "max_tokens": 4,
            "temperature": 0.0,
        },
    )
    assert response.status_code == 200, response.text
    assert engine.prefix_cache.hits > before


def test_the_cached_prefix_does_not_change_the_answer(
    client: TestClient, auth_headers: dict, engine
) -> None:
    """A request that reuses a cached prefix must produce the same answer."""

    from tinycode_engine.orchestration.request_pipeline import InferenceRequest

    prompt = LONG_PROMPT
    prompt_ids = engine.tokenize(prompt)

    def _run() -> str:
        request = InferenceRequest(
            request_id="compare",
            model_id="tinycode-test",
            prompt=prompt,
            prompt_ids=list(prompt_ids),
            max_tokens=6,
            temperature=0.0,
        )
        return engine.generate(request).text

    plain = _run()
    engine.cache_input(prompt)
    cached = _run()
    assert plain == cached
