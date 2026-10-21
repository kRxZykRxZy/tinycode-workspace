"""Tests of the OpenAI compatible HTTP API."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_models_requires_authentication(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_models_lists_the_loaded_models(client: TestClient, auth_headers: dict) -> None:
    response = client.get("/v1/models", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert any(model["id"] == "tinycode-test" for model in payload["data"])


def test_chat_completions_returns_a_response(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "tinycode-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert "content" in payload["choices"][0]["message"]
    assert payload["usage"]["total_tokens"] >= 0


def test_completions_returns_a_response(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/completions",
        headers=auth_headers,
        json={"model": "tinycode-test", "prompt": "hello", "max_tokens": 4},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "text_completion"
    assert "text" in payload["choices"][0]


def test_an_unknown_model_is_rejected(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_an_invalid_temperature_is_rejected(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "tinycode-test",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 5.0,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_tools_are_rejected_instead_of_being_ignored(
    client: TestClient, auth_headers: dict
) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "tinycode-test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        },
    )
    assert response.status_code == 400
    assert "tools" in response.json()["error"]["message"].lower()


def test_embeddings_return_vectors(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/embeddings",
        headers=auth_headers,
        json={"model": "tinycode-test", "input": "hello world"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    assert len(payload["data"][0]["embedding"]) > 0


def test_the_responses_endpoint_returns_a_response(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/responses",
        headers=auth_headers,
        json={"model": "tinycode-test", "input": "hello"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["output"][0]["content"][0]["text"] is not None


def test_health_endpoints(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_metrics_are_exposed(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "tinycode_requests_received" in response.text
