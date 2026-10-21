# TinyCode

A compact coding-model platform with two independently buildable
repositories:

| Repository | Responsibility |
| --- | --- |
| [`tinycode-model-builder/`](tinycode-model-builder) | Configuration, model construction, tokenization, dataset preparation, training, evaluation, checkpoints, the trainable adaptive-budget controller and the artifact export. |
| [`tinycode-inference-engine/`](tinycode-inference-engine) | Artifact loading, the custom inference runtime, the hardware backends, the adaptive computation budget at serving time, the scheduler, the caches, the OpenAI compatible API and the metrics. |

The two repositories share **no code**.  They integrate through the
versioned artifact contract in [`contracts/`](contracts) and through the
cross repository tests in [`integration-tests/`](integration-tests).

```
tinycode-workspace/
|
|-- tinycode-model-builder/      repository A
|-- tinycode-inference-engine/   repository B
|-- contracts/                   versioned specifications and golden fixtures
|-- integration-tests/           cross repository tests
|-- benchmarks/                  workspace level benchmarks
|-- docs/                        workspace level documentation
|-- scripts/                     workspace level helper scripts
|-- README.md
|-- WORKSPACE.md
```

## The two invariants

**The context length is 1024 tokens.**  It is hardcoded in both
repositories, cannot be changed by configuration, environment variable,
API field or hardware detection, and is enforced by the tests of both
repositories.  A prompt and its completion share that window.

**The parameter target is a user choice.**  The builder asks for it and
solves an architecture whose *actual* trainable parameter count matches
the target exactly.  The count is verified against an analytic formula
and recorded in the checkpoints, the reports and the artifact manifest.

## Getting started

```bash
# 1. Build a model
cd tinycode-model-builder
pip install -r requirements.txt
python main.py                     # interactive

# 2. Serve it
cd ../tinycode-inference-engine
pip install -r requirements.txt
python main.py serve --model ../tinycode-model-builder/artifacts/tinycode-latest
```

Then use any OpenAI compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-development-key")
print(client.chat.completions.create(
    model="tinycode-local",
    messages=[{"role": "user", "content": "Write a Python function that reverses a string."}],
    max_tokens=128,
).choices[0].message.content)
```

## Documentation

| Document | Content |
| --- | --- |
| [`WORKSPACE.md`](WORKSPACE.md) | The layout, the boundaries and the development workflow. |
| [`docs/architecture.md`](docs/architecture.md) | How the two repositories fit together. |
| [`docs/contracts.md`](docs/contracts.md) | The shared contract in detail. |
| [`docs/quickstart.md`](docs/quickstart.md) | A step by step walkthrough. |
| `tinycode-model-builder/docs/` | The builder internals. |
| `tinycode-inference-engine/docs/` | The engine internals, the API matrix and the backends. |

## Tests

```bash
# everything at once
pytest -q

# builder
cd tinycode-model-builder && pytest -q

# engine
cd tinycode-inference-engine && pytest -q

# cross repository
pytest -q integration-tests
```

Last run: **198 passed** (95 builder, 90 engine, 13 cross repository) in
70 s.  See [`docs/testing.md`](docs/testing.md) for the details, the
coverage and the bugs the suites found.

Hardware specific tests are skipped explicitly when the hardware or the
optional dependency is missing; a test that did not run is never reported
as passed.
