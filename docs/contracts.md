# The shared contract

The contract has three parts: the artifact manifest, the budget features
and the OpenAI compatibility behaviour.

## 1. The artifact manifest

The manifest is written by the builder and validated by the engine before
a single weight is loaded.  The canonical schema lives in
`contracts/artifact-manifest.v1.schema.json`.

| Section | Content |
| --- | --- |
| `artifact_format_version` | The version of the contract (`1.0.0`). |
| `model` | The name, the version and the architecture id. |
| `architecture` | The complete description of the decoder. |
| `parameters` | The requested target, the verified actual count and the breakdown. |
| `tokenizer` | The fingerprint, the vocabulary size and the file hashes. |
| `weights` | The format, the file name and the checksum. |
| `precisions` | The precisions the artifact can be used with. |
| `required_operators` | The operators a backend has to implement. |
| `controller` | The optional trainable budget controller. |
| `training`, `export` | The provenance metadata. |
| `compatibility` | The minimum engine version and the supported backends. |
| `files` | The SHA-256 and the size of every file. |

### The rules

* `architecture.context_length` **must** be 1024.  The engine rejects any
  other value.
* `parameters.requested_target` must equal `parameters.actual_trainable`.
* `tokenizer.fingerprint` must match the tokenizer file.  The engine
  refuses to load a model whose tokenizer does not match.
* The SHA-256 of every file in `files` must match.
* No secret is ever written to the manifest.

## 2. The budget features

`contracts/budget-features.v1.json` defines the 17 features of the
controller and their order.  The builder extracts them in
`budget/features.py`, the engine in `budget/controller.py`, and the
cross repository test asserts that the two vectors are identical.

A trained controller whose `feature_schema_version` does not match is
refused; the engine then uses the deterministic baseline policy.

## 3. The OpenAI compatibility behaviour

`contracts/openai-compat.v1.md` describes the endpoints, the response
shapes, the error shapes, the finish reasons, the streaming format, the
chat prompt format, the context budget policy and the usage accounting.

## The golden fixture

`contracts/golden/tinycode-golden` is a tiny, deterministic artifact.
Both repositories load it, which verifies that the manifest, the weight
layout and the tokenizer agree.

```bash
python scripts/make_golden_fixture.py
```

## Changing the contract

1. Change the specification in `contracts/`.
2. Bump the version: a major version for a breaking change, a minor
   version for a compatible addition.
3. Update the builder to emit the new version.
4. Update the engine to accept the new version.
5. Extend the cross repository tests.
6. Update the documentation and the compatibility matrix.
