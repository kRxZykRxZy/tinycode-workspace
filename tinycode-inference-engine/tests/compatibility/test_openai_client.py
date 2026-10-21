"""Tests of the OpenAI client compatibility.

The official ``openai`` client is used when it is installed; the tests
are skipped otherwise.  Nothing in the test suite contacts an external
model provider.
"""

from __future__ import annotations

import pytest

openai = pytest.importorskip("openai", reason="the openai client is not installed")


def test_the_official_client_can_talk_to_the_local_server(
    client, auth_headers: dict, monkeypatch
) -> None:
    """Configure the official client for the local server and query it."""

    transport = client.app.state  # keep a reference so the app is not collected

    from openai import OpenAI

    sdk_client = OpenAI(
        base_url="http://testserver/v1",
        api_key="test-key",
        http_client=_build_http_client(client),
    )
    response = sdk_client.chat.completions.create(
        model="tinycode-test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=4,
    )
    assert response.choices
    assert response.choices[0].message.role == "assistant"
    assert response.usage is not None
    assert transport is not None


def _build_http_client(test_client) -> object:
    """Adapt the FastAPI test client to the httpx interface."""

    import httpx

    class _Transport(httpx.BaseTransport):
        def __init__(self, client) -> None:
            self._client = client

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            response = self._client.request(
                request.method,
                request.url.raw_path.decode(),
                headers=dict(request.headers),
                content=request.content,
            )
            return httpx.Response(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=response.content,
                request=request,
            )

    return httpx.Client(transport=_Transport(test_client), base_url="http://testserver")
