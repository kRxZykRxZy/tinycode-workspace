# Architecture of the model builder

```
  application/     interactive CLI, menus, prompts, workflows
        |
        v
  configuration/   YAML loading, validation, parameter parsing,
        |         immutable context constraints
        v
  model/           architecture, factory (solver), causal model,
        |         parameter accounting
        v
  data/            The Stack client, filtering, deduplication,
        |         splitting, tokenisation, packing
        v
  training/        trainer, train step, optimizer, scheduler,
        |         precision, validation, stopping
        v
  checkpoint/      manager, serializer, integrity, resume, retention
        |
        v
  export/          safetensors / pytorch / gguf, manifest, verification
```

Supporting subsystems: `hardware/` (capability detection), `resources/`
(memory monitoring and recovery), `storage/` (local and Google Drive),
`budget/` (the trainable controller), `evaluation/`, `dashboard/`,
`telemetry/` and `utils/`.

## The architecture solver

`solve_architecture()` searches for a configuration whose analytic
parameter count equals the target exactly.

The parameter count of the decoder is

```
V * d                                  token embedding
+ (C * d)                              position embedding (learned only)
+ L * (2 * norm + 4 d^2 + bias_q)      attention of every layer
+ L * (2 d * F + bias_f)               feed forward of every layer
+ norm                                 final normalisation
+ (V * d + bias_h)                     output head (unless it is tied)
```

`d_ff` enters the formula linearly, so an exact solution exists whenever
the remaining budget divides evenly by the number of feed forward
coefficients.  The solver exploits that instead of adding ballast
parameters: every parameter of the model is used.

The formula is mirrored (independently) in the inference engine and the
cross repository integration test asserts that both agree.

## The immutable context length

`configuration/immutable_constraints.py` owns the constant 1024.  The
configuration loader rejects any file that contains a context length like
key, the model validates its configuration, the data pipeline packs
sequences of exactly 1024 tokens, the trainer rejects longer sequences and
the manifest records the value.

## Separation of concerns

The builder contains no serving runtime.  The inference engine contains no
training code.  The two repositories only share the versioned artifact
contract.
