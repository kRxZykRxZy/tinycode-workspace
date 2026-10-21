# The OpenAI compatible API

Base URL: `http://127.0.0.1:8000/v1`

## Authentication

The protected endpoints require an API key:

```
Authorization: Bearer <key>
```

The comparison is constant time (`hmac.compare_digest`) and the key is
never written to a log.  The default development key lives in
`config/security.yaml`; `TINYCODE_API_KEYS` adds or replaces keys at
runtime.  `/health` is not protected so that a supervisor can probe it.

## Endpoints

### `GET /v1/models`

Lists the loaded models with their ids and metadata.

### `POST /v1/completions`

Supported fields: `model`, `prompt`, `max_tokens`, `temperature`,
`top_p`, `top_k`, `stop`, `stream`, `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, `seed`, `echo`.

Only a single prompt per request is supported; a request with several
prompts is rejected instead of silently ignoring the extra ones.

### `POST /v1/chat/completions`

Supported fields: `model`, `messages`, `max_tokens` (or
`max_completion_tokens`), `temperature`, `top_p`, `top_k`, `stop`,
`stream`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`,
`seed`, `response_format`.

`tools`, `tool_choice` and `logprobs` are **not implemented**: they are
rejected with a clear error so that a client never believes a feature
worked when it did not.

The chat prompt format is part of the shared contract:

```
<|system|>\n...\n<|user|>\n...\n<|assistant|>\n
```

### `POST /v1/embeddings`

Returns the mean pooled final hidden states of the language model
backbone, L2 normalised.  This is useful for similarity search but it is
**not** a dedicated retrieval embedding model; the limitation is
documented in the compatibility matrix.

### `POST /v1/responses`

A documented subset: a single input, a text answer and usage reporting.
`store` and tool use are rejected.

### `POST /v1/cache/input`

Warms the runtime caches for an input.

```json
{"model": "tinycode-local", "input": "def add(a, b):", "prefix": true}
```

The response reports the number of tokens, the token ids and whether the
KV state was stored:

```json
{
  "object": "cache.entry",
  "cached": true,
  "model": "tinycode-local",
  "tokens": 6,
  "token_ids": [8, 27, 12, 29, 30, 31],
  "prefix_state_stored": true,
  "compute_ms": 4.182,
  "stats": {"token_cache": {}, "prefix_cache": {}}
}
```

A later request whose prompt starts with the cached input restores the
KV state and only computes the remaining tokens.  The cache key always
contains the model id, the model version and the tokenizer fingerprint,
so the state of an incompatible model is never reused.

A chat conversation can be cached instead of a plain text by passing
`messages`; it is rendered with the same chat prompt format as the
completion endpoints.

### `GET /v1/cache` and `DELETE /v1/cache`

Return the cache statistics and clear the caches.

### `GET /health`, `GET /health/ready`, `GET /metrics`

Liveness, readiness and the Prometheus metrics.

## Streaming

`stream: true` returns `text/event-stream` with OpenAI compatible
chunks:

```
data: {"id": "chatcmpl-...", "object": "chat.completion.chunk", ...}

data: [DONE]
```

The first chunk carries the assistant role, the following chunks carry the
content deltas and the last chunk carries the finish reason.  When the
client disconnects the request is cancelled, its KV cache is released and
the other requests are unaffected.

## Errors

Errors use the OpenAI compatible shape:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```

| Type | Status |
| --- | --- |
| `invalid_request_error` | 400 |
| `authentication_error` | 401 |
| `not_found_error` | 404 |
| `rate_limit_error` | 429 |
| `server_error` | 500 |
| `service_unavailable` | 503 |

## Limits

| Limit | Default | Configuration |
| --- | --- | --- |
| Request body | 1 MiB | `security.yaml: request_body_limit_bytes` |
| Prompt tokens | 1024 | fixed |
| Output tokens | 1024 | fixed by the context length |
| Stop sequences | 16 | `security.yaml` / safeguards |
| Rate limit | disabled | `security.yaml: rate_limit_per_minute` |
| Concurrent requests | 8 | `scheduler.yaml: max_active_requests` |
| Queued requests | 64 | `scheduler.yaml: queue_capacity` |
