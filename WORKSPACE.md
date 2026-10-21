# The workspace

## The two repositories

### Repository A — `tinycode-model-builder/`

Owns everything that happens *before* an artifact exists:

* the interactive configuration and the parameter target,
* the architecture solver and the parameter accounting,
* the tokenizer training,
* the dataset pipeline (The Stack),
* the training loop, the resource recovery and the checkpointing,
* the Google Drive integration for Colab,
* the evaluation and the reports,
* the trainable adaptive-budget controller,
* the artifact export and its verification.

It contains **no serving runtime**.  It produces the artifacts that
repository B serves.

### Repository B — `tinycode-inference-engine/`

Owns everything that happens *after* an artifact exists:

* the manifest validation and the artifact loading,
* the numpy forward pass with a KV cache,
* the hardware backends and the capability based selection,
* the optional native C++ kernels,
* the adaptive computation budget at serving time,
* the request scheduler, the admission control and the caches,
* the OpenAI compatible API and the streaming,
* the metrics and the diagnostics.

It contains **no training code** and does not import the builder.

## The boundaries

| Concern | Builder | Engine | Contract |
| --- | --- | --- | --- |
| Context length | `configuration/immutable_constraints.py` | `budget/request_analyzer.py`, `protocol/request_validation.py` | fixed at 1024 |
| Parameter count | `model/parameter_accounting.py` | (validated against the manifest) | `parameters.actual_trainable` |
| Architecture | `model/architecture.py` | `model/model_adapter.py` | `architecture` in the manifest |
| Tokenizer | `tokenizer/` | `model/tokenizer_adapter.py` | the tokenizer fingerprint |
| Budget features | `budget/features.py` | `budget/controller.py` | `budget_features.v1` |
| Chat prompt | `tokenizer/special_tokens.py` | `model/tokenizer_adapter.py` | `openai-compat.v1` |
| API | – | `api/`, `protocol/` | `openai-compat.v1` |

## The workspace directories

| Directory | Purpose |
| --- | --- |
| `contracts/` | The versioned specifications and the golden fixtures. Not a runtime dependency. |
| `integration-tests/` | The tests that import **both** repositories. |
| `benchmarks/` | Workspace level benchmarks that compare the two sides. |
| `docs/` | The workspace level documentation. |
| `scripts/` | Helper scripts (fixtures, bootstrap, contract verification). |

## The development workflow

1. Change the builder, run its tests, export an artifact.
2. Run the cross repository tests in `integration-tests/`.
3. Change the engine, run its tests and the integration tests again.
4. Update `contracts/` when the interface changes and bump the version.
5. Update the documentation and the compatibility matrix.

## The versioning

Each repository has its own version (`pyproject.toml`) and its own
release cycle.  The artifact contract has its own version
(`artifact_format_version`).  The engine refuses an artifact whose
contract version is newer than the one it implements; the builder records
the contract version it emits.
