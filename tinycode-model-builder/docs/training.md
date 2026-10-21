# Training and recovery

## The training loop

1. The trainer builds the model and verifies the parameter count.
2. The optimizer (AdamW by default) is created with two parameter groups:
   the matrices are decayed, the norms, the biases and the embeddings are
   not.
3. The scheduler warms up linearly and then decays with a cosine curve.
4. Every optimizer step processes the effective batch as micro batches,
   accumulates the gradients, unscales them, clips them and applies the
   update.
5. Every `eval_every` steps the model is evaluated on the validation
   shards; the validation tokens are counted separately.
6. Every `checkpoint_every` steps a checkpoint is written.

## Token accounting

`data/token_accounting.py` tracks the committed training tokens, the
validation tokens, the documents and the target.  Only tokens that were
successfully committed count towards the target; validation tokens never
do.  `TokenAccounting.ensure_target_reachable()` refuses a run whose
corpus is too small.

## Precision

`training/precision.py` resolves the precision:

* `auto` selects bfloat16 on a capable CUDA device, float16 otherwise and
  float32 on the CPU,
* an explicit precision that the device cannot support is downgraded with a
  recorded reason,
* the gradient scaler is only created for float16.

## Resource recovery

`resources/thresholds.py` classifies the memory usage:

| Usage | Level | Action |
| --- | --- | --- |
| below 75 % | normal | nothing |
| 75 - 89 % | reduced | the micro batch scale is halved |
| 90 - 94 % | caution | the loader workers are reduced |
| 95 % and above | critical | the training is paused safely |
| out of memory | oom | the micro batch is halved and the caches are released |

The recovery never changes the context length and never changes the
selected parameter target.  If the architecture does not fit into the
hardware the builder stops and explains the limitation instead of
retrying forever.

## The smoke test

Before a long run the builder executes a smoke test that verifies the
forward pass, the backward pass, the optimizer update, the checkpoint
round trip, the memory usage and the measured throughput.  A long run
additionally asks for confirmation.

## Resuming

A checkpoint stores the step, the committed training tokens, the model,
optimizer, scheduler and scaler state, the dataset position and the RNG
states.  `checkpoint/resume.py` restores them and the trainer continues
from the same point.
