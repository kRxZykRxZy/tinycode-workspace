# Quick start

## 1. Build a model

```bash
cd tinycode-model-builder
python -m venv .venv && . .venv/Scripts/activate      # Windows
pip install -r requirements.txt
python main.py
```

The builder is interactive.  A minimal path through the menu:

1. **Hardware capabilities** – check what the machine offers.
2. **Configure the model** – accept the default target of 10M parameters.
   The builder solves an architecture whose actual trainable parameter
   count equals the target exactly and prints the result.
3. **Train the tokenizer** – a small smoke tokenizer is created when no
   local corpus is available; train a real one on a directory of source
   files for a serious run.
4. **Prepare the dataset** – the builder validates the access to The
   Stack, downloads the shards, filters, deduplicates, splits, tokenises
   and packs the documents into 1024 token sequences.
5. **Train the model** – the smoke test runs first, then the trainer asks
   for the confirmation.  The default target is 200,000,000 training
   tokens; a much smaller target is fine for a first run.
6. **Export the artifact** – the builder writes `artifact.json`,
   `model.safetensors`, `model.pt` and the tokenizer into
   `artifacts/tinycode-latest` and verifies the result.

## 2. Serve the model

```bash
cd ../tinycode-inference-engine
pip install -r requirements.txt
python main.py serve --model ../tinycode-model-builder/artifacts/tinycode-latest
```

The server prints the endpoint, the loaded models, the selected backend
and the reason for that selection.

## 3. Talk to the server

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer local-development-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "tinycode-local", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}'
```

or with the official client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-development-key")
print(client.chat.completions.create(
    model="tinycode-local",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=32,
).choices[0].message.content)
```

## 4. Inspect the runtime

```bash
python main.py doctor     # the diagnostics of the environment
python main.py bench --model ../tinycode-model-builder/artifacts/tinycode-latest
curl http://127.0.0.1:8000/metrics
```

## 5. Run the tests

```bash
cd tinycode-model-builder && pytest -q
cd ../tinycode-inference-engine && pytest -q
cd .. && pytest -q integration-tests
```
## On Google Colab

```python
from google.colab import drive

drive.mount('/content/drive')
```

The builder detects the mount, mirrors every checkpoint to
`/content/drive/My Drive/tinycode` and verifies the copy.  The import of
`google.colab` only happens on the Colab path; outside Colab the builder
uses local storage and says so.
