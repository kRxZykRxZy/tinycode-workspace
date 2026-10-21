"""Benchmark runner: ties the evaluation components together."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

from ..utils.atomic_io import atomic_write_json
from .code_tasks import DEFAULT_CODE_TASKS, CodeTask, run_code_tasks
from .loss import LossResult
from .perplexity import compute_perplexity
from .reports import EvaluationReport

__all__ = ["BenchmarkRunner", "run_evaluation"]


class BenchmarkRunner:
    """Runs perplexity and coding task benchmarks against a model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        device: Any = None,
        tasks: Sequence[CodeTask] = DEFAULT_CODE_TASKS,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tasks = list(tasks)

    # ------------------------------------------------------------------
    def run_perplexity(self, batches: Any, max_batches: int = 32) -> LossResult:
        """Measure the perplexity on packed batches."""

        return compute_perplexity(self.model, batches, device=self.device, max_batches=max_batches)

    def run_code_tasks(self, generate: Callable[[CodeTask], str]) -> dict[str, Any]:
        """Run the structural coding benchmark."""

        return run_code_tasks(self.tasks, generate)

    def run(self, batches: Any = None, generate: Callable[[CodeTask], str] | None = None) -> EvaluationReport:
        """Run every enabled benchmark and return the report."""

        report = EvaluationReport(model_name=getattr(self.model, "config", None) and "tinycode")
        started = time.perf_counter()
        if batches is not None:
            perplexity = self.run_perplexity(batches)
            report.add("perplexity", perplexity.to_dict())
        if generate is not None:
            report.add("code_tasks", self.run_code_tasks(generate))
        report.add("runtime", {"elapsed_seconds": round(time.perf_counter() - started, 3)})
        return report


def run_evaluation(
    model: Any,
    tokenizer: Any,
    validation_shards: Sequence[str | Path],
    output_dir: str | Path,
    device: Any = None,
    generate: Callable[[CodeTask], str] | None = None,
    max_batches: int = 32,
) -> dict[str, Any]:
    """Run the evaluation and persist the report."""

    runner = BenchmarkRunner(model=model, tokenizer=tokenizer, device=device)

    def _batches():
        import numpy as np

        for shard in validation_shards:
            array = np.load(str(shard), mmap_mode="r")
            sequences = array.shape[0]
            for start in range(0, sequences - 1 + 1, 1):
                yield np.asarray(array[start : start + 1], dtype=np.int64)

    report = runner.run(batches=_batches(), generate=generate)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report.write(target / "evaluation.json")
    atomic_write_json(target / "evaluation.json", report.to_dict())
    return report.to_dict()
