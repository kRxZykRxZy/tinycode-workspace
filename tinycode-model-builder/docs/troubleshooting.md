# Troubleshooting

## "No supported architecture reaches exactly N trainable parameters"

The target cannot be reached with the supported architecture space.  The
builder prints the supported alternatives; choose a different target or
allow a reduced vocabulary.

## The vocabulary was reduced

A small target (for example 1M parameters) cannot hold a 98,304 token
embedding.  The builder reduces the vocabulary, says so and records the
vocabulary that was used in the manifest.

## The training run is very slow on the CPU

The T4 in Colab is the intended environment.  On the CPU the throughput
is a few tokens per second for a ten million parameter model; use a
smaller model or a smaller target for a smoke run.

## The dataset download fails

`python scripts/inspect_dataset.py --stack` checks the access.  A gated or
missing dataset is reported with the reason.  A local corpus directory
can be used instead.

## The dataset does not contain enough tokens

`TokenAccounting.ensure_target_reachable()` raises when the prepared
corpus is smaller than the requested target.  Prepare more data or
choose a smaller target; the target is never silently reduced.

## The Drive synchronisation failed

The local checkpoint is preserved.  Check that `/content/drive` is
mounted and that the account has free space.  The failure is reported in
the dashboard and in the logs.

## The checkpoint does not load

`python scripts/validate_artifact.py <dir>` verifies the artifact.  A
mismatch between the manifest and the files usually means that a file was
modified after the export.

## The dashboard is not reachable

The dashboard binds to `127.0.0.1` and requires the token that is printed
when it starts.  The token is also stored in `runs/dashboard_token.txt`.
A Cloudflare quick tunnel is optional and needs `cloudflared` to be
installed; without it the dashboard stays local.

## The training was interrupted

The last checkpoint is intact.  Restart the training workflow: the trainer
loads the newest checkpoint, restores the step, the committed tokens, the
optimizer, the scheduler and the RNG states and continues.
