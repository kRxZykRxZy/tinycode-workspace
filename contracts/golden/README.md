# Golden fixtures

This directory holds the fixtures that both repositories use to verify
their compatibility with the shared contract.

The golden artifact is a tiny, deterministic TinyCode model:

* vocabulary: 32 tokens,
* `d_model`: 16, 2 layers, 4 heads, `d_ff` 32,
* context length: 1024 (fixed),
* weights: generated from a fixed seed.

It is generated rather than committed so that the repository stays free
of binaries:

```bash
python scripts/make_golden_fixture.py
```

The generator writes:

```
golden/tinycode-golden/
    artifact.json          the manifest
    model.safetensors      the weights
    tokenizer.json         the tokenizer
    tokenizer_config.json  the tokenizer metadata
```

The integration tests create the fixture on demand when it is missing.
