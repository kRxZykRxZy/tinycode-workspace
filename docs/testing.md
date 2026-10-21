# Test results

The suites were executed on Windows with Python 3.14.3, `torch 2.14.1+cpu`,
numpy 2.4.6, tokenizers 0.23.1, safetensors, onnx, openvino 2026.4.1,
fastapi 0.136.3 and pytest 9.0.3.

| Suite | Command | Result |
| --- | --- | --- |
| Model builder | `cd tinycode-model-builder && python -m pytest` | **97 passed** (22 s) |
| Inference engine | `cd tinycode-inference-engine && python -m pytest` | **98 passed** (102 s) |
| Cross repository | `cd integration-tests && python -m pytest` | **15 passed** (23 s) |
| Everything | `python -m pytest` (workspace root) | **210 passed** (129 s) |
| Contracts | `python scripts/verify_contracts.py` | consistent (contract 1.0.0, 17 features) |

> Note: `pyproject.toml` sets `addopts = "-q"`.  Passing another `-q` on
> the command line turns it into `-qq`, which suppresses pytest's summary
> line (the tests still run and the exit code is 0).

## What is covered

**Model builder (97)**

* the parameter target parser (shorthand, explicit integers, rejects),
* the immutable 1024 token context (override attempts are detected),
* the parameter accounting (the analytic formula matches the live model,
  the solver reaches 1M/10M exactly),
* the tokenizer (train, save, load, round trip, fingerprint),
* the data primitives (filtering, deduplication, splitting, packing,
  token accounting),
* the checkpoint manager (atomic writes, integrity, retention, resume
  state),
* the budget controller (baseline policy, features, calibration),
* the manifest contract,
* the integration tests: a real training run (the optimizer changes the
  weights), the checkpoint round trip, the resume of a training run,
  the export and the verification of an artifact,
* the resource recovery policy.

**Inference engine (98)**

* the context invariant (1024 tokens, a 1025 token prompt is rejected,
  the KV cache refuses to grow past the limit, `max_tokens` is capped),
* the artifact validation (a valid artifact, a tampered file, a missing
  manifest, a context length override, an unknown architecture, a
  parameter mismatch, a missing fingerprint),
* the generation (deterministic greedy decoding, the length limit, the
  EOS token, cancellation, top-p, seeded sampling, the repetition
  penalty),
* the budget layer (the intake, the complexity estimate, the allocation,
  the server side limits, the verification, the entropy, the policies),
* the caches and the scheduler primitives (the LRU, the token cache key,
  the prefix cache key, the bounded queue, the priorities, the
  cancellation, the admission control),
* the adaptive budget policy (queue and memory pressure, the limits, the
  deadline),
* the scheduler (backpressure, the request level batching, the
  cancellation),
* the backends (the CPU backend is always available, the registry falls
  back to it, CUDA/Intel GPU/NEON are **not** claimed without the
  hardware),
* the **Intel graphics backend** (the graphics adapters are enumerated,
  the OpenVINO provider reports its devices, the GPU plugin is only
  claimed when it works, the GPU plugin is probed in a separate process,
  the backend refuses to run without a usable plugin and reports the
  device it really used),
* the HTTP API (authentication, the model list, the chat and the text
  completions, the unknown model, the invalid temperature, the rejected
  tools, the embeddings, the responses endpoint, the health and the
  metrics endpoints),
* the streaming (the SSE events, the role delta, the `[DONE]` marker),
* the **cache input route** (authentication, the caching, the empty
  input, the chat conversation, the statistics, the clearing, the reuse
  of a cached prefix and the unchanged answer),
* the OpenAI client compatibility (skipped when the `openai` package is
  not installed).

**Cross repository (15)**

* the context length is 1024 on both sides,
* the parameter formula is identical on both sides,
* the budget feature vector is identical on both sides,
* the chat prompt format is identical on both sides,
* the manifest matches the contract schema,
* the builder exports an artifact that the engine loads,
* the manifest validation rejects a tampered artifact,
* the logits of the torch model and of the numpy engine agree within
  1e-3 (measured: **3.7e-08**),
* the golden fixture loads in the engine,
* the engine serves without a trained controller,
* the **OpenVINO IR** that the builder exports runs in the engine and
  reproduces the PyTorch logits within 1e-3,
* the artifact carries the IR and the engine finds it.

## Bugs that the tests found and that are now fixed

1. **The engine's cached attention was not causal during the prefill.**
   Every query could attend to the future positions of the prompt.  The
   cross repository logits comparison caught it; the mask is now applied
   in `model/attention.py` and `runtime/operators.py`.
2. **The trainer passed the wrong object to the autocast helper**, so
   every training step crashed (`'_Plan' object has no attribute
   'use_amp'`).
3. **The resume did not restore the step and the committed token
   count.**
4. **`RunningMetrics` had no `restore` method**, so the resume crashed.
5. **The OOM recovery did not actually reduce the micro batch.**
6. **The output head registered the embedding as a child module**, so
   the exported `state_dict` contained the embedding twice and the
   parameter count of the artifact did not match the manifest.
7. **The export could produce an artifact without a tokenizer**, which
   neither repository accepts; the export now refuses it.
8. **The `content-length` header was compared as a string**, so every
   API request failed with a `TypeError`.
9. **A circular import** between `runtime`, `artifacts` and `model` broke
   the engine when it was imported from another repository.
10. **The weight layout of the test fixtures did not follow the torch
    convention** (`(out, in)`), which hid the parity of the forward
    pass.
11. **The OpenVINO integration used the removed `openvino.runtime`
    layout**, so the provider could not detect anything with OpenVINO
    2024+; both layouts are supported now.
12. **The Intel GPU plugin probe aborted the whole process** on a machine
    with a broken plugin DLL (Windows fatal exception 0xc0000139).  The
    probe now runs in a subprocess and a crash is reported as a failed
    probe.
13. **The Intel GPU backend compiled on the `GPU` device without
    checking that the plugin was usable**; it now refuses (or falls back
    to the CPU explicitly) and reports the device that was really used.

## Not covered / skipped

* The CUDA and ARM NEON backends are **skipped** on this machine (no
  NVIDIA GPU, no ARM host).  The tests assert that the backends are *not*
  claimed in that case; they do not assert that they work.
* The **Intel GPU acceleration itself could not be verified** here: the
  machine has an *AMD* Radeon 7500M/7600M (no Intel adapter) and the
  OpenVINO GPU plugin DLL cannot be loaded (`Cannot load library
  openvino_intel_gpu_plugin.dll: 127`).  The OpenVINO path itself is
  verified end to end (export → compile → infer → numerics → device
  reporting) on the CPU device.
* The native C++ kernels are **not built** here (no compiler is
  installed), so the engine used its numpy path.
* The `openai` client test is skipped when the package is missing; it
  ran here.
* No training run, no dataset download and no inference server session
  were executed: only the test suites.
