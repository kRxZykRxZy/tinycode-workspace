#!/usr/bin/env python
"""Streaming client for the local TinyCode server.

Usage:
    python examples/streaming_client.py
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def stream(url: str, payload: dict, api_key: str = "local-development-key") -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                print()
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    print(content, end="", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream from the local TinyCode server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="tinycode-local")
    parser.add_argument("--prompt", default="Explain what a Python generator is.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--api-key", default="local-development-key")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
    }
    stream(f"{args.base_url}/chat/completions", payload, args.api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
