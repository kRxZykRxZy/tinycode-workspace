# Deployment

## Local development (default)

```bash
python main.py serve --model ../tinycode-model-builder/artifacts/tinycode-latest
```

The server binds to `127.0.0.1:8000` and requires the API key from
`config/security.yaml` (`local-development-key` by default).

## Exposing the server to other machines

Remote access is an **explicit decision**:

1. change `host` in `config/server.yaml` (or pass `--host`),
2. replace the development key with a strong secret
   (`python -c "from tinycode_engine.security.api_keys import generate_api_key; print(generate_api_key())"`),
3. put the server behind a TLS terminating reverse proxy,
4. restrict the firewall to the addresses that may reach it,
5. keep the rate limit enabled.

The engine never exposes an unauthenticated service: the API key check
runs before the request is processed, and the key is compared in constant
time.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `TINYCODE_API_KEYS` | Comma separated list of accepted API keys. |
| `TINYCODE_MODEL` | Path of an artifact directory that is loaded at startup. |
| `TINYCODE_NATIVE_LIB` | Path of the compiled native library. |

## Health and metrics

* `GET /health` – liveness (unprotected).
* `GET /health/ready` – readiness; reports the loaded models and the
  selected backend.
* `GET /metrics` – Prometheus text format.

## Shutdown

`SIGINT`/`SIGTERM` stops the running requests, releases their KV caches
and shuts the backend down.  The shutdown report lists how many requests
were stopped.

## Running without an accelerator

The CPU path is a first class mode: it needs only numpy and the standard
library.  No compiler is required; the native kernels are optional.

## Capacity planning

The queue is bounded and the admission controller rejects work when the
limits are reached.  Size the machine for the concurrency that is needed:
each active request holds its own KV cache, the memory of which is

```
n_layers * 2 * n_heads * 1024 * head_dim * 4 bytes
```

For a 12 layer, 8 head, 512 dimension model that is about 100 MiB per
request at the full context length.
