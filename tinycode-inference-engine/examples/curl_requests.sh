#!/usr/bin/env bash
# Example requests against the local TinyCode server.
#
# The server binds to 127.0.0.1:8000 by default and requires the API key
# that is configured in config/security.yaml.

set -euo pipefail

BASE_URL="${TINYCODE_BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${TINYCODE_API_KEY:-local-development-key}"
MODEL="${TINYCODE_MODEL:-tinycode-local}"

echo "== model discovery =="
curl -sS "${BASE_URL}/v1/models" \
  -H "Authorization: Bearer ${API_KEY}" | python -m json.tool

echo
echo "== chat completion =="
curl -sS "${BASE_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Write a JavaScript function that adds two numbers.\"}],
    \"max_tokens\": 128
  }" | python -m json.tool

echo
echo "== text completion =="
curl -sS "${BASE_URL}/v1/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"def add(a, b):\",
    \"max_tokens\": 64
  }" | python -m json.tool

echo
echo "== streaming chat completion =="
curl -sS -N "${BASE_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Explain what a Python generator is.\"}],
    \"max_tokens\": 64,
    \"stream\": true
  }"

echo
echo "== health =="
curl -sS "${BASE_URL}/health/ready" | python -m json.tool
