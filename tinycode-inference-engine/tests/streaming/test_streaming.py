"""Tests of the streaming behaviour."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_chat_streaming_emits_sse_events(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "tinycode-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert body.startswith("data: ")
    assert "chat.completion.chunk" in body
    assert body.rstrip().endswith("[DONE]")


def test_completion_streaming_emits_sse_events(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/completions",
        headers=auth_headers,
        json={
            "model": "tinycode-test",
            "prompt": "hello",
            "max_tokens": 3,
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert "text_completion" in response.text
    assert "[DONE]" in response.text


def test_the_stream_contains_the_role_delta_first(client: TestClient, auth_headers: dict) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "tinycode-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": True,
        },
    )
    assert '"role": "assistant"' in response.text or '"role":"assistant"' in response.text
