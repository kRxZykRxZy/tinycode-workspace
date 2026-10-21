# Architecture of the inference engine

```
        HTTP request
             |
             v
   api/  (schemas, validation, authentication, SSE)
             |
             v
   orchestration/request_pipeline.py   <- the adaptive budget layer
      |            |            |
      v            v            v
 budget/       cache/       scheduling/
 (analyse,   (kv, prefix,   (queue, admission,
  allocate,    token)        cancellation,
  decide)                     backpressure)
             |            |
             v            v
        runtime/engine.py  <->  backends/  (cpu, cuda, intel_gpu, arm_neon)
             |
             v
        model/model_adapter.py  (numpy forward pass with a KV cache)
```

## Layers

| Layer | Module | Responsibility |
| --- | --- | --- |
| API | `api/` | Request schemas, validation, authentication, streaming, errors. |
| Protocol | `protocol/` | The OpenAI compatible payloads and their serialisation. |
| Orchestration | `orchestration/` | The request pipeline, planning, execution control, verification. |
| Budget | `budget/` | Request intake, complexity, allocation, decisions, verification, telemetry. |
| Scheduling | `scheduling/` | Queue, admission, cancellation, backpressure, request level batching. |
| Cache | `cache/` | KV cache, prefix cache, token cache, eviction. |
| Runtime | `runtime/` | The engine, operators, memory accounting, kernel dispatch. |
| Model | `model/` | The numpy forward pass, the tokenizer adapter. |
| Backends | `backends/` | The execution backends and the selection strategy. |
| Artifacts | `artifacts/` | Manifest validation, weight loading, integrity checks. |

## The forward pass

The decoder is implemented in `model/model_adapter.py`:

1. token embedding plus the position embeddings,
2. per layer: RMSNorm (or LayerNorm) -> causal attention -> residual,
   RMSNorm -> SiLU feed forward -> residual,
3. final norm and the output projection.

The KV cache stores the key and value vectors of every layer and is
bounded by the fixed 1024 token context.  A request that would exceed the
context is rejected, never silently truncated inside the model.

## Request level continuous batching

The TinyCode decoder has no batched attention path, so the scheduler
interleaves requests instead of batching their tensors: every round
advances each active request by one decoding step.  Each request owns its
own KV cache and its own state, so a cancelled request cannot affect the
others.

## Backend selection

`backends/registry.py` walks the priority list (CUDA, Intel GPU, ARM
NEON, CPU), initialises each candidate, runs a small allocation test and
selects the first backend that passes.  Every attempt is recorded so the
diagnostics can explain *why* a backend was not used.
