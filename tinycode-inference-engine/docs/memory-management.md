# Memory management

## The KV cache

`cache/kv_cache.py` stores the key and value vectors of every layer for
one request.

* The cache is bounded by the fixed 1024 token context; it is never
  allocated without an upper bound.
* The write position advances with the sequence length; a request that
  would exceed the context is rejected.
* The cache is released when the request finishes **and** when it is
  cancelled, so a disconnecting client cannot leak memory.

The memory of one cache is
`n_layers * 2 * n_heads * context_length * head_dim * sizeof(float32)`.

## The prefix cache

`cache/prefix_cache.py` caches the KV state of prompt prefixes.  The cache
key contains the model id, the model version and the tokenizer
fingerprint, so state is never reused across incompatible models or
tokenizers.  Prefixes shorter than `prefix_cache_min_tokens` are not
cached.

The prefix cache stores the *metadata* of a prefix by default.  Storing
and restoring the tensors of a KV cache is only correct when the tensors
belong to the same model instance, so the payload is optional and is only
used when the caller guarantees that.

## The token cache

`cache/token_cache.py` caches the token ids of prompts.  The key
contains the same identity components as the prefix cache.

## Eviction

`cache/eviction.py` implements an LRU cache with a configurable maximum
number of entries.  Eviction is bounded: the cache never grows without a
limit.

## Accounting

`runtime/memory.py` tracks the allocated and reserved bytes of the
runtime.  `memory/accounting.py` exposes it, `memory/pressure.py`
detects the memory pressure of the machine and `memory/cleanup.py`
releases the caches and the temporary buffers.

## Admission and backpressure

`scheduling/admission.py` refuses a request when the concurrency limit,
the queue capacity or the memory limit is reached.  The answer is a
`503` with a retry hint, not an unbounded queue.

`scheduling/backpressure.py` computes the pressure signal from the queue
depth, the concurrency and the memory pressure.  The policy when
pressure appears is:

1. stop admitting unnecessary work,
2. apply backpressure (the API answers with `503`),
3. cancel or reject the work according to the documented policy,
4. release the resources of the completed requests,
5. evict the eligible cache entries,
6. report the action in the metrics.

Model state is never migrated silently between backends.
