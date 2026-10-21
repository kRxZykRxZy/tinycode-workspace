# Troubleshooting

## The server starts but every request fails

Check `python main.py doctor`.  It prints:

* the detected devices and why a backend was rejected,
* the startup report with the models that were loaded.

A missing model is the most common cause: the engine then answers with
`503` and the message "No model is loaded into the server."

## "The artifact is not compatible with this engine"

The manifest is validated before the weights are loaded.  The usual
causes are:

* a different `artifact_format_version`,
* an `architecture_id` that the engine does not know,
* a `context_length` that is not 1024,
* a missing tokenizer fingerprint,
* a weight checksum that does not match.

`python ../tinycode-model-builder/scripts/validate_artifact.py <dir>`
verifies the artifact from the builder side.

## "The tokenizer does not match the fingerprint"

The tokenizer file was changed after the artifact was exported.  Re-export
the artifact; the engine refuses to serve a model whose tokenizer does
not match the manifest.

## The answers are empty

The model may be undertrained (a freshly initialised model produces
near-uniform logits).  Check the training report of the builder.  The
engine reports the token counts in the usage object, so an empty answer
with `completion_tokens: 0` points at the budget or the stop conditions
rather than at the model.

## The request is rejected with 503

The server is saturated: the concurrency limit or the queue capacity was
reached, or the machine is under memory pressure.  The response body
explains which limit was hit.  Increase `max_active_requests` and
`queue_capacity` in `config/scheduler.yaml` or add capacity.

## The prompt is truncated

The prompt is truncated from the left so that the instruction at the end
of a chat prompt survives.  The prompt budget and the output budget share
the same 1024 token window: a 900 token prompt leaves 124 tokens for the
completion.  Reduce `max_tokens` to get a longer prompt.

## Streaming stops without `[DONE]`

The client disconnected.  The request is cancelled and its resources are
released.  Check the client side logs and the `x-request-id` header.

## The native kernels are not used

The library is searched in `native/build`, `native/build/lib` and
`src/tinycode_engine/runtime/_native`; `TINYCODE_NATIVE_LIB` overrides the
search.  `python main.py doctor` prints whether the native library was
loaded.  The engine works without it.

## CUDA is not selected

The CUDA backend needs PyTorch *and* a usable device.  Without both, the
registry falls back to the CPU path and records the reason in the
selection report.  The engine never claims GPU acceleration while the
work runs on the CPU.

## Intel GPU is not selected

The Intel backend needs Intel graphics *and* a verified OpenVINO GPU
plugin *and* an OpenVINO IR of the model.  The builder does not export an
IR yet, so the backend reports itself as unavailable.  This limitation is
documented in `docs/backends.md`.
