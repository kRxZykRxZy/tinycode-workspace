"""Policy datasets built from recorded request outcomes.

The dataset is the bridge between production telemetry and offline
policy improvement.  It never stores request content: only the derived
features and the measured outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .features import FEATURE_SCHEMA_VERSION, BudgetFeatures
from .training_objectives import ControllerTargets

__all__ = ["PolicySample", "PolicyDataset", "build_policy_dataset", "save_policy_dataset", "load_policy_dataset"]


@dataclass
class PolicySample:
    """One training example for the controller."""

    features: BudgetFeatures
    targets: ControllerTargets

    def to_dict(self) -> dict[str, Any]:
        return {"features": self.features.to_dict(), "targets": self.targets.to_dict()}


@dataclass
class PolicyDataset:
    """A collection of policy samples."""

    samples: list[PolicySample] = field(default_factory=list)
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def __len__(self) -> int:
        return len(self.samples)

    def feature_matrix(self) -> list[list[float]]:
        return [sample.features.to_vector() for sample in self.samples]

    def target_matrix(self) -> list[list[float]]:
        return [
            [
                sample.targets.budget_multiplier,
                float(sample.targets.plan),
                float(sample.targets.verify),
                float(sample.targets.stop),
            ]
            for sample in self.samples
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.feature_schema_version,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def build_policy_dataset(records: Iterable[dict[str, Any]]) -> PolicyDataset:
    """Build a policy dataset from recorded request records.

    Expected record fields:

    ``features``     - raw feature values (see :class:`BudgetFeatures`)
    ``budget_multiplier`` - the multiplier that produced the best outcome
    ``plan`` / ``verify`` / ``stop`` - binary decisions
    ``quality``      - measured answer quality in ``[0, 1]``
    """

    dataset = PolicyDataset()
    for record in records:
        raw_features = record.get("features") or {}
        features = BudgetFeatures(
            prompt_tokens=int(raw_features.get("prompt_tokens", 0)),
            requested_output_tokens=int(raw_features.get("requested_output_tokens", 0)),
            task_category=str(raw_features.get("task_category", "code")),
            complexity_score=float(raw_features.get("complexity_score", 0.0)),
            uncertainty_score=float(raw_features.get("uncertainty_score", 0.0)),
            wants_verification=bool(raw_features.get("wants_verification", False)),
            latency_target_ms=(
                float(raw_features["latency_target_ms"])
                if raw_features.get("latency_target_ms") is not None
                else None
            ),
            queue_pressure=float(raw_features.get("queue_pressure", 0.0)),
            memory_pressure=float(raw_features.get("memory_pressure", 0.0)),
            backend_tokens_per_second=float(raw_features.get("backend_tokens_per_second", 64.0)),
            plan_steps=int(raw_features.get("plan_steps", 0)),
            computation_budget=int(raw_features.get("computation_budget", 1024)),
        )
        targets = ControllerTargets(
            budget_multiplier=float(record.get("budget_multiplier", 1.0)),
            plan=int(record.get("plan", 0)),
            verify=int(record.get("verify", 0)),
            stop=int(record.get("stop", 0)),
            quality=float(record.get("quality", 0.0)),
        )
        dataset.samples.append(PolicySample(features=features, targets=targets))
    return dataset


def save_policy_dataset(dataset: PolicyDataset, path: str | Path) -> Path:
    """Persist a policy dataset as JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dataset.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_policy_dataset(path: str | Path) -> PolicyDataset:
    """Load a policy dataset from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset = PolicyDataset(feature_schema_version=payload.get("feature_schema_version", FEATURE_SCHEMA_VERSION))
    for item in payload.get("samples", []):
        raw_features = item.get("features", {})
        raw_targets = item.get("targets", {})
        dataset.samples.append(
            PolicySample(
                features=BudgetFeatures(
                    prompt_tokens=int(raw_features.get("prompt_tokens", 0)),
                    requested_output_tokens=int(raw_features.get("requested_output_tokens", 0)),
                    task_category=str(raw_features.get("task_category", "code")),
                    complexity_score=float(raw_features.get("complexity_score", 0.0)),
                    uncertainty_score=float(raw_features.get("uncertainty_score", 0.0)),
                    wants_verification=bool(raw_features.get("wants_verification", False)),
                    latency_target_ms=raw_features.get("latency_target_ms"),
                    queue_pressure=float(raw_features.get("queue_pressure", 0.0)),
                    memory_pressure=float(raw_features.get("memory_pressure", 0.0)),
                    backend_tokens_per_second=float(raw_features.get("backend_tokens_per_second", 64.0)),
                    plan_steps=int(raw_features.get("plan_steps", 0)),
                    computation_budget=int(raw_features.get("computation_budget", 1024)),
                ),
                targets=ControllerTargets(
                    budget_multiplier=float(raw_targets.get("budget_multiplier", 1.0)),
                    plan=int(raw_targets.get("plan", 0)),
                    verify=int(raw_targets.get("verify", 0)),
                    stop=int(raw_targets.get("stop", 0)),
                    quality=float(raw_targets.get("quality", 0.0)),
                ),
            )
        )
    return dataset


def split_dataset(
    dataset: PolicyDataset, validation_fraction: float = 0.2
) -> tuple[PolicyDataset, PolicyDataset]:
    """Split a dataset deterministically into train and validation parts."""

    total = len(dataset.samples)
    cut = int(total * (1.0 - validation_fraction))
    train = PolicyDataset(
        samples=dataset.samples[:cut], feature_schema_version=dataset.feature_schema_version
    )
    validation = PolicyDataset(
        samples=dataset.samples[cut:], feature_schema_version=dataset.feature_schema_version
    )
    return train, validation
