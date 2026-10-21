# Shared contracts

This directory holds the **specifications** that the two repositories
agree on.  It is deliberately not a third runtime repository: it contains
no Python package that the builder or the engine has to install.

| File | Purpose |
| --- | --- |
| `artifact-manifest.v1.schema.json` | The JSON schema of the artifact manifest. |
| `budget-features.v1.json` | The feature vector of the adaptive budget controller. |
| `openai-compat.v1.md` | The behaviour of the OpenAI compatible API. |
| `golden/` | The golden fixtures that both repositories use. |

Both repositories implement these specifications independently:

* the builder writes the manifest (`tinycode_builder.export.artifact_manifest`),
* the engine validates it (`tinycode_engine.artifacts.manifest`),
* the builder extracts the budget features
  (`tinycode_builder.budget.features`),
* the engine extracts the same features (`tinycode_engine.budget.controller`),
* the builder renders the chat prompt
  (`tinycode_builder.tokenizer.special_tokens.chat_prompt`),
* the engine renders the same prompt
  (`tinycode_engine.model.tokenizer_adapter.chat_prompt`).

The cross repository tests in `integration-tests/` assert that the two
implementations agree.

## Versioning

The contract version is `1.0.0`.  A change that breaks an existing
artifact requires a new major version; a compatible addition requires a
minor version.  The engine refuses an artifact whose contract version is
newer than the one it implements, and the builder records the version it
emits.

## Golden fixtures

`golden/` contains a tiny artifact that both repositories load.  It is
generated deterministically:

```bash
python scripts/make_golden_fixture.py
```

The integration tests generate it on demand when it is missing, so the
fixtures never have to be committed as binaries.
