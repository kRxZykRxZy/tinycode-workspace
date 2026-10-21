"""Training of the adaptive budget controller.

The controller is a small MLP that predicts

* the budget multiplier (regression),
* whether planning is warranted (binary),
* whether verification is warranted (binary).

It is trained on recorded request outcomes.  Training the controller is
optional: the inference engine falls back to the deterministic baseline
policy when no controller artifact is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .controller import TrainableBudgetController, save_controller
from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .policy_dataset import PolicyDataset, split_dataset

__all__ = ["ControllerTrainingResult", "train_controller"]


@dataclass
class ControllerTrainingResult:
    """Outcome of a controller training run."""

    controller: TrainableBudgetController
    epochs: int = 0
    train_loss: float = 0.0
    validation_loss: float = 0.0
    samples: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "samples": self.samples,
            "history": list(self.history[-20:]),
        }


def _make_model(hidden_sizes: list[int], feature_count: int) -> Any:
    import torch
    from torch import nn

    layers: list[nn.Module] = []
    previous = feature_count
    for size in hidden_sizes:
        layers.append(nn.Linear(previous, size))
        layers.append(nn.Tanh())
        previous = size
    layers.append(nn.Linear(previous, 4))
    return nn.Sequential(*layers)


def train_controller(
    dataset: PolicyDataset,
    hidden_sizes: list[int] | None = None,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    output_dir: str | Path | None = None,
) -> ControllerTrainingResult:
    """Train the budget controller on a policy dataset."""

    import torch
    from torch import nn

    hidden_sizes = hidden_sizes or [64, 32]
    train_set, validation_set = split_dataset(dataset, validation_fraction)
    feature_count = len(FEATURE_NAMES)

    model = _make_model(hidden_sizes, feature_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    def _features(samples: PolicyDataset) -> "torch.Tensor":
        return torch.tensor(samples.feature_matrix(), dtype=torch.float32)

    def _targets(samples: PolicyDataset) -> "torch.Tensor":
        return torch.tensor(samples.target_matrix(), dtype=torch.float32)

    history: list[dict[str, Any]] = []
    train_loss = 0.0
    validation_loss = 0.0

    if len(train_set) > 0:
        features = _features(train_set)
        targets = _targets(train_set)
        for epoch in range(max(1, epochs)):
            optimizer.zero_grad()
            predictions = model(features)
            budget_loss = criterion(predictions[:, 0], targets[:, 0])
            plan_loss = nn.functional.binary_cross_entropy_with_logits(
                predictions[:, 1], targets[:, 1]
            )
            verify_loss = nn.functional.binary_cross_entropy_with_logits(
                predictions[:, 2], targets[:, 2]
            )
            loss = budget_loss + 0.5 * (plan_loss + verify_loss)
            loss.backward()
            optimizer.step()
            train_loss = float(loss.item())
            history.append({"epoch": epoch + 1, "loss": train_loss})

        model.eval()
        with torch.no_grad():
            if len(validation_set) > 0:
                validation_predictions = model(_features(validation_set))
                validation_targets = _targets(validation_set)
                validation_loss = float(
                    criterion(validation_predictions[:, 0], validation_targets[:, 0]).item()
                )
            else:
                validation_loss = 0.0

    controller = TrainableBudgetController(
        {},
        config={
            "hidden_sizes": hidden_sizes,
            "feature_names": list(FEATURE_NAMES),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
    )
    controller.load_weights(model.state_dict())

    if output_dir is not None:
        save_controller(controller, output_dir)

    return ControllerTrainingResult(
        controller=controller,
        epochs=max(1, epochs) if len(train_set) else 0,
        train_loss=train_loss,
        validation_loss=validation_loss,
        samples=len(dataset),
        history=history,
    )
