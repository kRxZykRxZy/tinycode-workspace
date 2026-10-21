# Checkpoints and Google Drive

## The checkpoint contents

| Field | Meaning |
| --- | --- |
| `model_state` | the weights |
| `optimizer_state`, `scheduler_state`, `scaler_state` | the training state |
| `step` | the optimizer step |
| `committed_training_tokens` | the tokens that were committed |
| `validation_tokens` | the validation tokens (never part of the target) |
| `dataset_position` | the position in the shard stream |
| `rng_state` | the Python, NumPy and PyTorch RNG states |
| `architecture` | the architecture and the verified parameter count |
| `parameter_target`, `actual_parameters` | the selected target and the measured count |
| `tokenizer_fingerprint` | the fingerprint of the tokenizer |
| `metrics` | the training metrics |
| `recovery_state` | the resource recovery state |

A JSON manifest next to the checkpoint records the same metadata plus the
SHA-256 of every file.

## Atomic writes and integrity

`checkpoint/serializer.py` writes the checkpoint into a temporary file and
renames it afterwards, so an interrupted write never destroys the
previous known-good checkpoint.  `checkpoint/integrity.py` verifies the
hashes of the files, and `checkpoint/manager.py` refuses to load a
checkpoint whose hash does not match.

## Retention

`checkpoint/retention.py` keeps the last *n* checkpoints and optionally
every *n*-th one.  The deletion only happens after a new checkpoint was
written successfully.

## Google Drive

On Colab the builder mounts Drive with Colab's own module:

```python
from google.colab import drive

drive.mount('/content/drive')
```

The import only happens on the Colab path.  Outside Colab the builder
uses local storage and says so explicitly.

`storage/atomic_copy.py` copies a checkpoint to Drive atomically and
verifies the copy.  `storage/synchronization.py` reports whether the copy
is intact; a failed synchronisation never removes the local checkpoint.

The Drive mount is detected, so an existing mount is not remounted.
