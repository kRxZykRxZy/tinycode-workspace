#!/usr/bin/env python
"""Minimal client for the local TinyCode server.

Usage:
    python examples/python_client.py
    python examples/python_client.py --prompt "Write a quicksort in Python"
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def post(url: str, payload: dict, api_key: str = "local-development-key") -> dict:
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
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the local TinyCode server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="tinycode-local")
    parser.add_argument("--prompt", default="Write a JavaScript function that adds two numbers.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--api-key", default="local-development-key")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    result = post(f"{args.base_url}/chat/completions", payload, args.api_key)
    print(result["choices"][0]["message"]["content"])
    print(json.dumps(result.get("usage", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
