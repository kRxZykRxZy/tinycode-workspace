# The OpenAI compatibility contract (v1)

This document describes the behaviour that both repositories have to
agree on.  It is a specification, not code: the two repositories
implement it independently.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/models` | The loaded models. |
| POST | `/v1/completions` | Text completion. |
| POST | `/v1/chat/completions` | Chat completion. |
| POST | `/v1/embeddings` | Embeddings. |
| POST | `/v1/responses` | The (partial) Responses API. |
| GET | `/health` | Liveness. |
| GET | `/health/ready` | Readiness. |
| GET | `/metrics` | Prometheus metrics. |

## Authentication

```
Authorization: Bearer <api key>
```

The comparison is constant time.  The endpoints that are not protected
are the health endpoints only.

## Response shapes

Chat completion:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "tinycode-local",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "..."}, "logprobs": null, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
}
```

Text completion:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1700000000,
  "model": "tinycode-local",
  "choices": [{"index": 0, "text": "...", "logprobs": null, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
}
```

Error:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```

## Finish reasons

`stop`, `length`, `cancelled`, `content_filter` (reserved), `tool_calls`
(reserved).

## Streaming

```
data: {"id": "...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}

data: {"id": "...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}

data: {"id": "...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

## The chat prompt format

The engine renders a conversation into the prompt with the special tokens
of the tokenizer:

```
<|system|>
{system}

<|user|>
{user}

<|assistant|>
{answer}

<|assistant|>
```

The generator prompt ends with `<|assistant|>\n` so that the model
continues the assistant turn.

## The context budget

The prompt and the completion share the same 1024 token window.  A
request with a 900 token prompt and a 300 token completion does not fit:
the engine truncates the prompt (from the left) and reduces the output
budget to the remaining tokens.

## Usage accounting

`prompt_tokens` counts the tokens that were fed into the model,
`completion_tokens` counts the tokens that were generated and
`total_tokens` is their sum.  The internal computation of the adaptive
budget layer is reported separately in the telemetry, not in the usage
object.
