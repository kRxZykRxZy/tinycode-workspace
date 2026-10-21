# TinyCode Model Builder

Builds, trains, evaluates, checkpoints and exports the TinyCode coding
model together with its optional trainable adaptive-budget controller.

* The **parameter target is chosen interactively** and the architecture is
  solved so that the actual trainable parameter count matches it
  **exactly**.
* The **context length is fixed at 1024 tokens**.  It is not configurable
  through YAML, environment variables, API fields or hardware detection.
* The corpus is **The Stack** from Hugging Face.
* On Google Colab the checkpoints are mirrored to **Google Drive** using
  Colab's own `google.colab` module.

The repository produces the artifacts that the separate
**tinycode-inference-engine** repository serves.

---

## Quick start

```bash
pip install -r requirements.txt

python main.py
```

The builder is interactive.  The menu offers:

| Option | Purpose |
| --- | --- |
| 1 | Inspect the hardware capabilities (GPU, RAM, disk, Colab). |
| 2 | Configure the model: parameter target, vocabulary, architecture. |
| 3 | Prepare the dataset: The Stack download, filtering, packing. |
| 4 | Train the coding tokenizer. |
| 5 | Train the model (smoke test first, then the real run). |
| 6 | Evaluate the model (perplexity and structural coding tasks). |
| 7 | Export the artifact for the inference engine. |
| 8 | Train the optional adaptive budget controller. |
| 9 | Start the authenticated training dashboard. |
| 10 | Show the environment diagnostics. |

Every workflow can also be run without the menu:

```bash
python main.py --workflow configure
python main.py --workflow data
python main.py --workflow train
```

## The parameter target

The builder asks for the target number of trainable parameters:

```
  Supported parameter targets:
    - 1M
    - 10M (default)
    - 25M
    - 50M
    - 100M
    - 500M
    - 1B
```

The solver searches a structured space of decoder architectures
(vocabulary, embedding tying, positional encoding, normalisation type,
bias configuration) times `(d_model, n_layers, d_ff)` and keeps the
candidate whose **analytic** parameter count equals the target exactly.
The measured count of the live model is verified against the formula, and
the builder refuses to continue when they disagree.

For very small targets the vocabulary is reduced (the preferred 98,304
tokens cannot fit a one million parameter model); the builder says so and
records the vocabulary that was used.

## The fixed context length

`tinycode_builder.configuration.immutable_constraints.CONTEXT_LENGTH` is
1024 and cannot be changed.  The loader rejects any configuration file
that tries to set a context length like key, the training sequences are
exactly 1024 tokens, the token packing produces 1024 token sequences and
the manifest records `context_length: 1024`.

## Training

The default target is 200,000,000 **training** tokens; validation tokens
are counted separately and never count towards the target.  Before a long
run the builder executes a smoke test that verifies the forward pass, the
backward pass, the optimizer update, the checkpoint round trip, the
memory usage and the measured throughput.  A long run asks for
confirmation.

The trainer supports AdamW, warmup with a cosine schedule, weight decay,
gradient clipping, mixed precision, gradient accumulation, memory aware
micro batches, validation, checkpointing, resume, graceful interruption
and a final evaluation.

## Resource safety

| Usage | Level | Action |
| --- | --- | --- |
| below 75 % | normal | - |
| 75 - 89 % | reduced | the micro batch is reduced |
| 90 - 94 % | caution | the loader workers are reduced |
| 95 % and above | critical | the training is paused safely |
| out of memory | oom | immediate recovery |

Recovery never changes the context length and never changes the selected
parameter target.

## Checkpoints

A checkpoint contains the weights, the model configuration, the parameter
target and the verified count, the optimizer, scheduler and scaler state,
the step, the committed training tokens, the dataset position, the RNG
states, the tokenizer fingerprint, the metrics, the recovery state and
the integrity metadata.  Writes are atomic and the previous known-good
checkpoint is preserved.

On Colab the checkpoint is mirrored to Google Drive and the copy is
verified before the synchronisation is reported as successful.

## The dashboard

```bash
python main.py --workflow dashboard
```

The dashboard shows the model and the parameter count, the token target
and the progress, the throughput, the estimated completion time, the loss
and the validation loss, the optimizer step, the learning rate, the GPU
and CPU utilisation, the VRAM and the RAM, the disk space, the dataset
progress, the checkpoint state, the Drive synchronisation and the recent
logs.

The controls are: request a safe checkpoint, pause, resume, stop
gracefully and view the logs.  The dashboard is always authenticated,
independently of the (optional) Cloudflare quick tunnel, and it never
exposes arbitrary shell execution.  Training continues when the dashboard
or the tunnel disconnects.

## Export

```bash
python scripts/validate_artifact.py artifacts/tinycode-latest
```

The exported directory contains `artifact.json` (the versioned manifest),
`model.safetensors`, `model.pt`, the tokenizer files and the optional
controller.  The manifest records the requested parameter target, the
verified actual count, the fixed context length, the vocabulary, the
tokenizer fingerprint, the weight hashes, the required operators, the
controller metadata and the compatibility information.

GGUF export is implemented for the supported subset (RMSNorm, learned
positions, F32/F16) and raises a clear error for everything else.

An **OpenVINO IR** export (ONNX → IR) is also available: it is what the
inference engine's Intel graphics backend executes.  The export is
verified numerically against the PyTorch model before it is accepted.

```bash
python -c "from tinycode_builder.export.pipeline import export_artifact; ..."
```

## Documentation

* `docs/architecture.md` – the components of the builder.
* `docs/training.md` – the training loop and the recovery policy.
* `docs/dataset.md` – the data pipeline and the packing.
* `docs/checkpoints.md` – the checkpoint format and the Drive sync.
* `docs/budget-controller.md` – the trainable budget controller.
* `docs/troubleshooting.md` – common problems.

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```
