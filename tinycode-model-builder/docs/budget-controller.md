# The adaptive budget controller

The controller decides how much computation a request receives.  It is
*trainable* in the model builder and *executable* in the inference engine;
the two sides share the feature schema `budget_features.v1`.

## The interface

```python
decision = controller.decide(features, requested_tokens)
```

`features` is a `BudgetFeatures` object, `decision` is a
`ControllerDecision` with the recommended policy, the generation tokens,
the number of internal steps, the verification passes, the planning flag,
the confidence and a rationale.

## The features

The feature vector and its order are part of the contract:

| # | Feature |
| --- | --- |
| 0 | `prompt_tokens_normalised` |
| 1 | `requested_output_normalised` |
| 2 - 6 | the task category one-hot flags |
| 7 | `complexity_score` |
| 8 | `uncertainty_score` |
| 9 | `wants_verification` |
| 10 | `has_latency_target` |
| 11 | `latency_target_normalised` |
| 12 | `queue_pressure` |
| 13 | `memory_pressure` |
| 14 | `backend_speed` |
| 15 | `plan_steps_normalised` |
| 16 | `computation_budget_normalised` |

A controller whose feature schema does not match is refused by the engine;
it then uses the deterministic baseline policy.

## The deterministic baseline

`budget/controller.py` contains the baseline policy that is always
available:

* complexity below 0.25 -> `minimal`,
* below 0.55 -> `standard`,
* above -> `extended`,
* verification when the request asks for it or the uncertainty is high.

The baseline is transparent and inspectable; it is the reference the
trained controller is compared against.

## Training the controller

```bash
python main.py --workflow controller
```

The controller is trained on *recorded request outcomes*
(`runs/policy_records.json`), never on raw user content.  The objective
combines a regression on the log budget multiplier with a binary
cross entropy on the planning and verification decisions.

`budget/evaluation.py` compares the trained controller with the baseline:
the mean absolute error of the budget, the bias, the decision accuracy,
the mean allocated tokens and the compute efficiency score.

The artefacts are stored as `controller.pt` (the weights) and
`controller.json` (the feature schema, the hidden sizes and the version).
The export copies them into the artifact directory and the manifest
references them.

## Calibration

`budget/calibration.py` fits a two parameter logistic rescaling of a
heuristic signal.  The calibration metadata states explicitly that a
calibrated signal is still not a measure of factual correctness.
