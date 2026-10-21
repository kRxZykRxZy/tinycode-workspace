# The adaptive computation budget

The budget layer decides **how much computation** a request receives.  It
never changes the context length: the context stays at 1024 tokens in
every policy.

## The stages

| # | Stage | Module |
| --- | --- | --- |
| 1 | Request intake | `budget/request_analyzer.py` |
| 2 | Task and complexity analysis | `budget/complexity_estimator.py` |
| 3 | Initial budget allocation | `budget/budget_allocator.py`, `budget/policy.py` |
| 4 | Planning and decomposition | `orchestration/planner.py` |
| 5 | Model execution | `orchestration/execution_controller.py` |
| 6 | Confidence and uncertainty | `budget/confidence.py` |
| 7 | Adaptive computation decision | `budget/scheduler.py` |
| 8 | Verification | `budget/verification.py`, `orchestration/verification_pipeline.py` |
| 9 | Budget aware finalisation | `budget/stopping.py` |
| 10 | Feedback and calibration | `budget/telemetry.py`, `budget/calibration` (builder side) |

## The budget object

`budget/controller.py` defines the `RequestBudget`: the request id, the
estimated complexity, the requested latency and compute budget, the
maximum generation tokens, the maximum internal steps, the verification
allowance, the deadline, the memory estimate, the selected policy, the
allocated budget, the consumed budget and the stop reason.

The budgets have different units and are never interchangeable:

* **context length** – the maximum input sequence (1024 tokens),
* **output budget** – the maximum generated tokens,
* **computation budget** – the maximum work the system may spend,
* **verification budget** – the work for the checks and additional passes,
* **latency budget** – the deadline,
* **memory budget** – the resources the admission policy allows.

## Policies

| Policy | Output tokens | Steps | Planning | Verification |
| --- | --- | --- | --- | --- |
| `minimal` | 32 – 128 | 1 | no | no |
| `standard` | 64 – 384 | 2 | no | no |
| `extended` | 128 – 768 | 4 | yes | no |
| `verification` | 96 – 640 | 4 | yes | 1 pass |

## The controller

The engine always has a deterministic baseline controller
(`budget/controller.py`).  A trained controller can be attached from the
artifact; when it is missing or its feature schema does not match
(`budget_features.v1`), the baseline is used and the engine keeps
serving.  The learned controller is never a single point of failure.

## Uncertainty signals

`budget/confidence.py` computes explicitly named heuristics: the mean
entropy of the generated tokens, the ratio of low probability tokens, the
agreement between attempts and the verification outcome.  They are *not*
calibrated probabilities of correctness, and the documentation says so.
The builder side fits a calibration model (`budget/calibration.py`) that
rescales a signal onto a probability; it never claims the signal itself
becomes a correctness measure.

## Verification

The verifiers are structural: JSON syntax, balanced code fences, a
non-empty answer, the presence of a code block.  **Generated code is
never executed** – that would require an isolated sandbox which the
engine deliberately does not provide.  A response that passed a
formatting check is not reported as correct.

## Telemetry

`budget/telemetry.py` records aggregate measurements only: the task
category, the allocated and consumed budget, the latency, the token
usage, the verification outcome.  No request content is stored, and the
records are never used to train on user requests without an explicit
policy.

## Safeguards

`budget/safeguards.py` clamps every user supplied limit to the server
side limits.  A user cannot ask for more tokens, steps or latency than
the server allows.
