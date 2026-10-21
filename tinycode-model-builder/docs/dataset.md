# The dataset pipeline

```
  The Stack (Hugging Face)
        |
        v
  schema detection      -> content / language / path / repo
        |
        v
  filtering             -> length, language, printable ratio
        |
        v
  deduplication         -> exact hash + MinHash near duplicates
        |
        v
  normalisation         -> line endings, trailing whitespace
        |
        v
  splitting             -> deterministic hash based train/validation split
        |
        v
  tokenisation          -> the coding tokenizer
        |
        v
  packing               -> sequences of exactly 1024 tokens
        |
        v
  shards                -> .npy files + dataset_report.json
```

## Access

`data/access_validation.py` checks that the dataset can be listed before
a long download starts.  The builder tries `bigcode/the-stack-smol` first
and falls back to the other The Stack mirrors.  A local corpus directory
can be selected explicitly; nothing else is mixed in automatically.

## Filtering

`data/filtering.py` drops the documents that are too short or too long,
that do not match the language filter and that are not mostly printable.
The statistics are part of the dataset report.

## Deduplication

`data/deduplication.py` removes the exact duplicates with a 64 bit hash of
the normalised content and the near duplicates with a 64 permutation
MinHash signature over 5 token shingles.

## Splitting

`data/splitting.py` assigns every document to the train or the validation
split with a salted hash of its identity.  The split is reproducible and
no document can appear in both splits.

## Packing

`data/packing.py` concatenates the tokenised documents with an end of text
token between them and cuts the stream into blocks of exactly 1024
tokens.  The number of documents per block is recorded so that the
training report can show how much context mixing occurred.

## Accounting

The training tokens and the validation tokens are tracked separately.
The training report contains the token counts, the filter and
deduplication statistics and the elapsed time.
