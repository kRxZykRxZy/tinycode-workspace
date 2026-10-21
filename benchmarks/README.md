# Workspace level benchmarks.
#
# The scripts delegate to the benchmarks of the inference engine and add
# the cross repository checks.

Run everything:

    python benchmarks/run_all.py --model ../tinycode-model-builder/artifacts/tinycode-latest

| Benchmark | Measures |
| --- | --- |
| `throughput.py` | The generated tokens per second. |
| `latency.py` | The end to end latency and the time to the first token. |
| `backend_comparison.py` | Which backends could be initialised and how long they need. |
| `budget_quality.py` | The effect of the budget policy on the token usage and the latency. |
| `memory_usage.py` | The weight memory and the process memory. |

Every script prints JSON so that the results can be compared over time.
The numbers are **measured**, never estimated: a benchmark that could not
run reports the error instead of a default value.
