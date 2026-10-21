# Architecture of the workspace

## The end to end path

```
  tinycode-model-builder                      tinycode-inference-engine
  -----------------------                    --------------------------

  configuration  ->  parameter target
        |
        v
  model.factory  ->  architecture with the exact parameter count
        |
        v
  tokenizer      ->  coding oriented byte level BPE
        |
        v
  data           ->  The Stack -> filtered -> packed 1024 token sequences
        |
        v
  training       ->  real optimizer steps, checkpoints, resume
        |
        v
  export         ->  artifact.json + model.safetensors + tokenizer
        |
        +------------------ versioned contract -----------------+
                                                                   |
                                                                   v
                                              artifacts.loader -> manifest validation
                                                                   |
                                                                   v
                                              backends.registry -> capability based selection
                                                                   |
                                                                   v
                                              model.model_adapter -> numpy forward pass
                                                                   |
                                                                   v
                                              budget/ -> allocation, decision, verification
                                                                   |
                                                                   v
                                              api/ -> OpenAI compatible responses
```

## Why two repositories

The builder and the engine have very different dependencies and very
different lifecycles:

* the builder needs PyTorch, the dataset tooling and the training
  environment (a Colab T4 in the intended workflow);
* the engine needs numpy, a tokenizer and an HTTP server, and it has to
  run on a laptop or on a small server.

Merging them would force every serving machine to install the training
stack.  Keeping them separate also means that the model format, the
tokenizer and the budget policy can evolve independently as long as the
contract holds.

## The contract

The contract is a specification, not code:

* `contracts/artifact-manifest.v1.schema.json` – the manifest,
* `contracts/budget-features.v1.json` – the controller features,
* `contracts/openai-compat.v1.md` – the API behaviour,
* `contracts/golden/` – a tiny artifact that both sides load.

Both repositories implement the specifications independently and the
integration tests assert that the implementations agree.

## The fixed context length

The value 1024 appears in four places, and every one of them is covered
by a test:

| Repository | Location | Test |
| --- | --- | --- |
| Builder | `configuration/immutable_constraints.py` | `tests/unit/test_immutable_constraints.py` |
| Builder | `data/packing.py` | `tests/unit/test_data_primitives.py` |
| Engine | `budget/request_analyzer.py` | `tests/unit/test_context_invariant.py` |
| Engine | `cache/kv_cache.py` | `tests/unit/test_context_invariant.py` |

The cross repository test `integration-tests/test_contracts.py` asserts
that both constants are the same.

## The parameter accounting

The builder owns the formula (`model/parameter_accounting.py`) and the
solver (`model/factory.py`).  The engine does not count the parameters
itself: it trusts the manifest and warns when the weight file contains a
different number of tensors.  The cross repository test recomputes the
formula independently and compares it with the builder.
