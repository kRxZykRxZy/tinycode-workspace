"""A small, self contained coding task benchmark.

The benchmark measures whether the model produces a response that contains
the expected structural elements of a solution.  It is deliberately simple
and *not* comparable to frontier coding benchmarks; it exists so that the
builder can measure whether additional computation improves the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = ["CodeTask", "DEFAULT_CODE_TASKS", "score_completion", "run_code_tasks"]


@dataclass
class CodeTask:
    """A single coding prompt with structural expectations."""

    name: str
    prompt: str
    required_fragments: tuple[str, ...] = ()
    forbidden_fragments: tuple[str, ...] = ()
    category: str = "code"
    max_tokens: int = 128

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "required_fragments": list(self.required_fragments),
            "forbidden_fragments": list(self.forbidden_fragments),
            "category": self.category,
            "max_tokens": self.max_tokens,
        }


DEFAULT_CODE_TASKS: tuple[CodeTask, ...] = (
    CodeTask(
        name="add_function_python",
        prompt=(
            "Write a Python function `def add(a, b):` that returns the sum of two "
            "numbers.\n\n```python\n"
        ),
        required_fragments=("def add", "return"),
        category="code",
    ),
    CodeTask(
        name="reverse_string_python",
        prompt=(
            "Write a Python function `def reverse(text):` that returns the reversed "
            "string.\n\n```python\n"
        ),
        required_fragments=("def reverse", "return", "[::-1]"),
        category="code",
    ),
    CodeTask(
        name="fizzbuzz_python",
        prompt="Write a Python function `def fizzbuzz(n):` that returns the classic sequence.\n\n```python\n",
        required_fragments=("def fizzbuzz", "FizzBuzz"),
        category="code",
    ),
    CodeTask(
        name="fetch_js",
        prompt=(
            "Write a JavaScript function `async function fetchJson(url)` that fetches "
            "JSON from a URL.\n\n```javascript\n"
        ),
        required_fragments=("async", "fetch", "json"),
        category="code",
    ),
    CodeTask(
        name="explain_loop",
        prompt="Explain in one paragraph what a Python `for` loop does.",
        required_fragments=("loop",),
        category="explain",
    ),
    CodeTask(
        name="json_output",
        prompt='Return a JSON object with the keys "name" and "age".',
        required_fragments=("{", "name", "age"),
        category="structured",
    ),
)


def score_completion(task: CodeTask, completion: str) -> dict[str, Any]:
    """Score a completion against the structural expectations of a task."""

    text = completion or ""
    lowered = text.lower()
    missing = [fragment for fragment in task.required_fragments if fragment.lower() not in lowered]
    forbidden_hit = [
        fragment for fragment in task.forbidden_fragments if fragment.lower() in lowered
    ]
    passed = not missing and not forbidden_hit
    return {
        "task": task.name,
        "category": task.category,
        "passed": passed,
        "missing_fragments": missing,
        "forbidden_fragments": forbidden_hit,
        "completion_length": len(text),
        "score": 1.0 if passed else 0.0,
    }


def run_code_tasks(
    tasks: Sequence[CodeTask],
    generate: Callable[[CodeTask], str],
) -> dict[str, Any]:
    """Run the benchmark using a generation callable.

    Args:
        tasks: the tasks to run.
        generate: callable that receives a task and returns the completion.

    Returns:
        A report with the per-task results and the aggregate score.
    """

    results = []
    for task in tasks:
        try:
            completion = generate(task)
        except Exception as exc:
            results.append(
                {
                    "task": task.name,
                    "category": task.category,
                    "passed": False,
                    "error": str(exc),
                    "score": 0.0,
                }
            )
            continue
        results.append(score_completion(task, completion))

    passed = sum(1 for result in results if result.get("passed"))
    total = len(results) or 1
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        category = result.get("category", "code")
        entry = by_category.setdefault(category, {"total": 0, "passed": 0})
        entry["total"] += 1
        if result.get("passed"):
            entry["passed"] += 1

    return {
        "tasks": len(results),
        "passed": passed,
        "score": passed / total,
        "by_category": by_category,
        "results": results,
    }
