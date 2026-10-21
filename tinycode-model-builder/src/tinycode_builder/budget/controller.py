"""The adaptive budget controller.

Two implementations live here:

``BaselineBudgetController``
    A deterministic, transparent policy.  It is always available and is
    the fallback used by the inference engine when no trained controller
    is present.

``TrainableBudgetController``
    A small MLP that predicts the budget multiplier and the planning /
    verification decisions from the feature vector.  It is optional: the
    system stays functional without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, BudgetFeatures
from .policy import BUILTIN_POLICIES, BudgetPolicy, get_policy

__all__ = [
    "ControllerDecision",
    "BudgetController",
    "BaselineBudgetController",
    "TrainableBudgetController",
    "save_controller",
    "load_controller",
]


@dataclass
class ControllerDecision:
    """What the controller recommends for a request."""

    policy: str = "standard"
    generation_tokens: int = 256
    internal_steps: int = 1
    verification_passes: int = 0
    allow_planning: bool = False
    confidence: float = 0.0
    rationale: str = ""
    source: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "generation_tokens": self.generation_tokens,
            "internal_steps": self.internal_steps,
            "verification_passes": self.verification_passes,
            "allow_planning": self.allow_planning,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source": self.source,
        }


class BudgetController:
    """Interface shared by the baseline and the trainable controller."""

    version = "1.0.0"
    feature_schema_version = FEATURE_SCHEMA_VERSION

    def decide(self, features: BudgetFeatures, requested_tokens: int) -> ControllerDecision:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError


class BaselineBudgetController(BudgetController):
    """Deterministic policy based on explicit rules.

    The rules are intentionally simple and inspectable:

    * complexity below 0.25 -> ``minimal``
    * complexity below 0.55 -> ``standard``
    * complexity below 0.8  -> ``extended``
    * otherwise -> ``extended`` with verification when the request asks
      for it or the estimated uncertainty is high.
    """

    version = "1.0.0"

    def decide(self, features: BudgetFeatures, requested_tokens: int) -> ControllerDecision:
        complexity = max(0.0, min(1.0, features.complexity_score))
        uncertainty = max(0.0, min(1.0, features.uncertainty_score))

        if complexity < 0.25:
            policy_name = "minimal"
        elif complexity < 0.55:
            policy_name = "standard"
        else:
            policy_name = "extended"

        wants_verification = bool(features.wants_verification) or uncertainty >= 0.6
        if wants_verification and complexity >= 0.55:
            policy_name = "verification"

        policy = get_policy(policy_name)
        tokens = policy.clamp_tokens(requested_tokens)

        # A latency target shrinks the budget; a generous target grows it.
        if features.latency_target_ms is not None:
            tokens = int(tokens * min(1.0, features.latency_target_ms / 30_000.0))
            tokens = max(policy.min_generation_tokens, tokens)

        verification_passes = policy.max_verification_passes if policy.allow_verification else 0
        internal_steps = min(policy.max_internal_steps, max(1, features.plan_steps + 1))

        return ControllerDecision(
            policy=policy.name,
            generation_tokens=tokens,
            internal_steps=internal_steps,
            verification_passes=verification_passes,
            allow_planning=policy.allow_planning and complexity >= 0.55,
            confidence=round(1.0 - uncertainty, 4),
            rationale=(
                f"complexity={complexity:.2f} uncertainty={uncertainty:.2f} "
                f"policy={policy.name}"
            ),
            source="baseline",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "baseline",
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "policies": {name: policy.to_dict() for name, policy in BUILTIN_POLICIES.items()},
        }


class TrainableBudgetController(BudgetController):
    """MLP controller trained on recorded request outcomes."""

    version = "1.0.0"

    def __init__(self, weights: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.weights = weights
        self.hidden_sizes = list(self.config.get("hidden_sizes", [64, 32]))
        self.feature_names = list(self.config.get("feature_names", FEATURE_NAMES))
        self.feature_schema_version = str(
            self.config.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        )
        self._model = None

    # ------------------------------------------------------------------
    @property
    def model(self) -> Any:
        """Lazily build the torch module."""

        if self._model is None:
            self._model = self._build_module()
            self._model.eval()
        return self._model

    def _build_module(self) -> Any:
        import torch
        from torch import nn

        layers: list[nn.Module] = []
        previous = len(self.feature_names)
        for size in self.hidden_sizes:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.Tanh())
            previous = size
        layers.append(nn.Linear(previous, 4))
        return nn.Sequential(*layers)

    def load_weights(self, state_dict: dict[str, Any]) -> None:
        """Load the trained weights into the controller."""

        self.weights = state_dict
        self._model = None

    # ------------------------------------------------------------------
    def predict(self, features: BudgetFeatures) -> dict[str, float]:
        """Return the raw controller outputs for a feature vector."""

        import torch

        vector = torch.tensor([features.to_vector()], dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(vector)
        values = outputs[0].tolist()
        return {
            "budget_multiplier": float(values[0]),
            "planning_logit": float(values[1]),
            "verification_logit": float(values[2]),
            "stop_logit": float(values[3]),
        }

    def decide(self, features: BudgetFeatures, requested_tokens: int) -> ControllerDecision:
        """Combine the learned output with the deterministic bounds."""

        try:
            raw = self.predict(features)
        except Exception:
            fallback = BaselineBudgetController().decide(features, requested_tokens)
            fallback.rationale = "trained controller unavailable; used the baseline policy"
            return fallback

        multiplier = max(0.25, min(4.0, raw["budget_multiplier"]))
        policy = get_policy("standard")
        tokens = policy.clamp_tokens(int(requested_tokens * multiplier))

        planning = raw["planning_logit"] > 0.0
        verification = raw["verification_logit"] > 0.0
        internal_steps = 1 + (1 if planning else 0) + (1 if verification else 0)

        return ControllerDecision(
            policy="verification" if verification else ("extended" if planning else "standard"),
            generation_tokens=tokens,
            internal_steps=min(internal_steps, policy.max_internal_steps),
            verification_passes=1 if verification else 0,
            allow_planning=planning,
            confidence=max(0.0, min(1.0, 1.0 - features.uncertainty_score)),
            rationale=(
                f"trained controller: multiplier={multiplier:.2f} "
                f"plan={planning} verify={verification}"
            ),
            source="trained",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "trained",
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "hidden_sizes": self.hidden_sizes,
            "feature_names": self.feature_names,
        }


def save_controller(
    controller: TrainableBudgetController,
    directory: str | Path,
) -> dict[str, Any]:
    """Persist a trained controller and its metadata."""

    import torch

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    state = controller.weights
    if controller._model is not None:
        state = controller._model.state_dict()
    torch.save(state, target / "controller.pt")

    metadata = {
        "version": controller.version,
        "feature_schema_version": controller.feature_schema_version,
        "feature_names": controller.feature_names,
        "hidden_sizes": controller.hidden_sizes,
        "config": controller.config,
    }
    (target / "controller.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"directory": str(target), **metadata}


def load_controller(directory: str | Path) -> TrainableBudgetController:
    """Load a trained controller from disk."""

    import torch

    directory = Path(directory)
    metadata_path = directory / "controller.json"
    weights_path = directory / "controller.pt"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"No controller artifact found in {directory}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    controller = TrainableBudgetController(state, config=metadata)
    controller.load_weights(state)
    return controller
