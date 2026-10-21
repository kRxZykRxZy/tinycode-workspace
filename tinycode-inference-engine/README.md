# TinyCode Inference Engine

A custom inference runtime for TinyCode models with an OpenAI compatible
HTTP API.  The engine loads the artifacts produced by the
**tinycode-model-builder** repository and serves them locally.

* No PyTorch dependency: the forward pass is implemented with numpy (and
  optional native C++ kernels).
* The context length is **fixed at 1024 tokens** and cannot be changed by
  configuration, environment variable or API field.
* Capability based backend selection: CUDA, a verified Intel graphics
  provider, ARM NEON, and a dependable CPU fallback.
* An adaptive computation budget layer that decides *how much* compute a
  request receives, independently of the context length.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Point the engine at an artifact exported by the model builder
python main.py serve --model ../tinycode-model-builder/artifacts/tinycode-latest

# 2. Talk to it with any OpenAI compatible client
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer local-development-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "tinycode-local", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}'
```

The official OpenAI client works against the local server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-development-key")
response = client.chat.completions.create(
    model="tinycode-local",
    messages=[{"role": "user", "content": "Write a JavaScript function that adds two numbers."}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

## Commands

| Command | Purpose |
| --- | --- |
| `python main.py serve` | Start the OpenAI compatible server (default). |
| `python main.py doctor` | Print the environment diagnostics and the startup report. |
| `python main.py models` | List the configured model paths. |
| `python main.py tokenize "text"` | Tokenise a text with the model tokenizer. |
| `python main.py bench --model <dir>` | Run a small local latency/throughput benchmark. |

Useful flags: `--model <artifact dir>`, `--host`, `--port`, `--api-key`,
`--config <directory>`.

## Configuration

The YAML files in `config/` are merged on startup:

| File | Contents |
| --- | --- |
| `server.yaml` | Bind address and port (default `127.0.0.1:8000`). |
| `runtime.yaml` | Artifact paths, default model alias, backend preference. |
| `scheduler.yaml` | Queue capacity and concurrency limits. |
| `budget_policy.yaml` | Server side limits of the computation budget. |
| `cache.yaml` | Cache sizes and the eviction policy. |
| `hardware.yaml` | Backend preference (`auto`, `cuda`, `intel_gpu`, `arm_neon`, `cpu`). |
| `security.yaml` | API keys, rate limit, request body limit. |

The API key can also be supplied through `TINYCODE_API_KEYS`
(comma separated) and the model directory through `TINYCODE_MODEL`.

## API compatibility matrix

| Feature | Status | Notes |
| --- | --- | --- |
| `GET /v1/models` | implemented | Lists the loaded models. |
| `POST /v1/completions` | implemented | Single prompt, `echo`, `stop`, `stream`. |
| `POST /v1/chat/completions` | implemented | `messages`, `max_tokens`, `temperature`, `top_p`, `stop`, `stream`. |
| `POST /v1/embeddings` | implemented | Mean pooled backbone states, **not** a dedicated retrieval model. |
| `POST /v1/responses` | partial | Single input, text output, usage. No tools, no store, no background mode. |
| `POST /v1/cache/input` | implemented | Warms the token and the prefix cache for an input. |
| `GET /v1/cache`, `DELETE /v1/cache` | implemented | The cache statistics and clearing the caches. |
| `GET /health`, `GET /health/ready` | implemented | Liveness and readiness. |
| `GET /metrics` | implemented | Prometheus text format. |
| SSE streaming | implemented | `data:` events terminated with `data: [DONE]`. |
| `tools` / `tool_choice` | not implemented | Rejected with a clear error. |
| `logprobs` | not implemented | Rejected with a clear error. |
| `n > 1` completions | not implemented | Rejected with a clear error. |
| Multiple prompts per request | not implemented | Rejected with a clear error. |
| `response_format` | partial | `text` and `json_object` are accepted; the JSON check is structural. |
| `seed` | implemented | Deterministic sampling for a given seed. |

## Hardware backends

| Backend | Requirement | Status |
| --- | --- | --- |
| CPU | numpy | Always available, first class. |
| CUDA | PyTorch with a usable CUDA device | Implemented; verified only where a GPU exists. |
| Intel GPU | Intel graphics **and** a verified OpenVINO GPU plugin **and** an OpenVINO IR of the model | Implemented and verified end to end (export → compile → infer → device reporting); the GPU acceleration itself was **not** verifiable on the development machine (no Intel adapter, GPU plugin DLL unloadable). |
| ARM NEON | ARM CPU with NEON and the native kernels | Implemented; falls back to the portable CPU path when the native library is not built. |

`python main.py doctor` prints which backends were detected, which were
verified and why a backend was rejected.

## Native kernels

```bash
cmake -S native -B native/build -DTINYCODE_BUILD_TESTS=ON
cmake --build native/build --config Release
ctest --test-dir native/build
```

The shared library is picked up automatically (`native/build`,
`native/build/lib` or `TINYCODE_NATIVE_LIB`).  Without it the engine uses
the numpy implementations; no compiler is required to run the engine.

## Documentation

* `docs/architecture.md` – the components of the engine.
* `docs/backends.md` – backend detection and selection.
* `docs/api.md` – the endpoints and their limitations.
* `docs/adaptive-budget.md` – the ten stages of the budget layer.
* `docs/memory-management.md` – caches, KV cache and memory accounting.
* `docs/deployment.md` – hosting the engine safely.
* `docs/troubleshooting.md` – common problems.

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```

Hardware specific tests are skipped explicitly when the hardware or the
optional dependency is missing.
