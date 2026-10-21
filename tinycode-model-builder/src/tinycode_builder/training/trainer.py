"""The training engine.

The trainer owns the model, the optimizer, the schedule, the data stream
and the checkpoint policy.  It is deliberately independent from the
interactive application layer so that it can be driven from a script or a
test.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ..configuration.immutable_constraints import CONTEXT_LENGTH
from ..utils.errors import TrainingError
from ..utils.reproducibility import capture_rng_state, seed_everything
from .metrics import RunningMetrics
from .optimizer import build_optimizer
from .precision import PrecisionPlan, build_scaler, resolve_precision
from .scheduler import SchedulerConfig, build_scheduler
from .stopping import StopReason, StoppingCriteria, TrainingStopper
from .train_step import run_train_step
from .validation import evaluate_loss, iter_validation_batches

__all__ = ["Trainer", "TrainingResult", "TrainingEvent"]


@dataclass
class TrainingEvent:
    """A progress event emitted by the trainer."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.payload}


@dataclass
class TrainingResult:
    """Summary of a completed training run."""

    steps: int = 0
    committed_training_tokens: int = 0
    validation_tokens: int = 0
    final_loss: float = 0.0
    final_validation_loss: float = 0.0
    stop_reason: str = ""
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0
    checkpoints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "committed_training_tokens": self.committed_training_tokens,
            "validation_tokens": self.validation_tokens,
            "final_loss": self.final_loss,
            "final_validation_loss": self.final_validation_loss,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "checkpoints": list(self.checkpoints),
        }


class Trainer:
    """Trains a TinyCode model on packed 1024 token sequences."""

    def __init__(
        self,
        config: dict[str, Any],
        model_config: Any,
        run_dir: str | Path,
        train_shards: list[str | Path] | None = None,
        validation_shards: list[str | Path] | None = None,
        tokenizer_fingerprint: str = "",
        device: str | None = None,
        checkpoint_manager: Any = None,
        synchronizer: Any = None,
        on_event: Callable[[TrainingEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.train_shards = [Path(path) for path in (train_shards or [])]
        self.validation_shards = [Path(path) for path in (validation_shards or [])]
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.checkpoint_manager = checkpoint_manager
        self.synchronizer = synchronizer
        self.on_event = on_event

        training_cfg = dict(config.get("training", {}) or {})
        self.training_config = training_cfg
        self.target_tokens = int(training_cfg.get("target_tokens", 200_000_000))
        self.batch_size = max(1, int(training_cfg.get("batch_size", 8)))
        self.micro_batch_size = max(1, int(training_cfg.get("micro_batch_size", 1)))
        self.accumulation_steps = max(1, self.batch_size // self.micro_batch_size)
        self.grad_clip = float(training_cfg.get("grad_clip", 1.0))
        self.log_every = max(1, int(training_cfg.get("log_every", 20)))
        self.eval_every = max(1, int(training_cfg.get("eval_every", 500)))
        self.checkpoint_every = max(1, int(training_cfg.get("checkpoint_every", 1000)))
        self.seed = int(training_cfg.get("seed", 1337))

        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.model: Any = None
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.scaler: Any = None
        self.precision: PrecisionPlan = resolve_precision(
            str(training_cfg.get("precision", "auto")),
            self.device,
            cuda_supports_bf16=self._cuda_supports_bf16(),
        )
        self.metrics = RunningMetrics()
        self.stopper = TrainingStopper(
            StoppingCriteria(
                target_tokens=self.target_tokens,
                max_steps=int(training_cfg.get("max_steps", 0)),
                max_hours=float(training_cfg.get("max_hours", 0.0)),
            )
        )
        self.committed_training_tokens = 0
        self.validation_tokens = 0
        self.step = 0
        self.dataset_position = 0
        self.stop_requested = False
        self.recovery: Any = None
        self.architecture_description: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover
            return False

    @staticmethod
    def _cuda_supports_bf16() -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            return bool(torch.cuda.get_device_properties(0).major >= 8)
        except Exception:  # pragma: no cover
            return False

    def build(self) -> "Trainer":
        """Create the model, optimizer and scheduler."""

        import torch

        from ..model.causal_model import build_model
        from ..model.parameter_accounting import verify_parameter_count

        seed_everything(self.seed)
        self.model = build_model(self.model_config)
        actual = verify_parameter_count(self.model, self.model_config)
        self.architecture_description = self.model_config.to_dict()
        self.architecture_description["actual_parameters"] = actual
        self.model.to(self.device)

        self.optimizer = build_optimizer(
            self.model,
            name=str(self.training_config.get("optimizer", "adamw")),
            learning_rate=float(self.training_config.get("learning_rate", 3e-4)),
            weight_decay=float(self.training_config.get("weight_decay", 0.1)),
            betas=(
                float(self.training_config.get("beta1", 0.9)),
                float(self.training_config.get("beta2", 0.95)),
            ),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            SchedulerConfig(
                warmup_steps=int(self.training_config.get("warmup_steps", 500)),
                max_steps=max(1, self._estimate_total_steps()),
                min_lr=float(self.training_config.get("min_learning_rate", 3e-5)),
                max_lr=float(self.training_config.get("learning_rate", 3e-4)),
            ),
        )
        self.scaler = build_scaler(self.precision)
        self._emit("built", {"parameters": actual, "device": self.device, "precision": self.precision.to_dict()})
        return self

    def _estimate_total_steps(self) -> int:
        tokens_per_step = self.batch_size * CONTEXT_LENGTH
        if tokens_per_step <= 0:
            return 1
        return max(1, self.target_tokens // tokens_per_step)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def _load_shard(self, path: Path) -> Any:
        import numpy as np

        return np.load(str(path), mmap_mode="r")

    def iter_training_batches(self) -> Iterator[Any]:
        """Yield batches of packed sequences."""

        import numpy as np

        if not self.train_shards:
            raise TrainingError(
                "No training shards are available. Prepare the dataset before training."
            )
        for shard_path in self.train_shards:
            try:
                array = self._load_shard(shard_path)
            except Exception as exc:
                raise TrainingError(f"Failed to read shard {shard_path}: {exc}") from exc
            sequences = array.shape[0]
            index = 0
            while index + self.batch_size <= sequences:
                window = np.asarray(array[index : index + self.batch_size], dtype=np.int64)
                index += self.batch_size
                self.dataset_position = index
                yield window

    def iter_validation_batches(self, limit: int = 20) -> Iterator[Any]:
        """Yield validation batches."""

        import numpy as np

        for shard_path in self.validation_shards:
            array = self._load_shard(shard_path)
            sequences = array.shape[0]
            for start in range(0, sequences - self.batch_size + 1, self.batch_size):
                window = np.asarray(array[start : start + self.batch_size], dtype=np.int64)
                yield window
                limit -= 1
                if limit <= 0:
                    return

    # ------------------------------------------------------------------
    # smoke test
    # ------------------------------------------------------------------
    def smoke_test(self) -> dict[str, Any]:
        """Verify forward pass, backward pass, optimizer update and checkpoint IO."""

        import torch

        if self.model is None:
            self.build()

        report: dict[str, Any] = {}
        sequence_length = min(CONTEXT_LENGTH, 32)
        input_ids = torch.randint(
            0, max(1, self.model_config.vocab_size - 1), (1, sequence_length), device=self.device
        )
        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]

        started = time.perf_counter()
        output = self.model(input_ids, labels=labels)
        if output.loss is None:  # pragma: no cover
            raise TrainingError("The model did not produce a loss.")
        report["forward_loss"] = float(output.loss.item())
        report["forward_seconds"] = round(time.perf_counter() - started, 4)

        self.optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            self.grad_clip,
        )
        report["gradient_norm"] = float(grad_norm)
        self.optimizer.step()
        report["optimizer_step"] = "ok"

        if self.checkpoint_manager is not None:
            try:
                info = self.checkpoint_manager.save(
                    {
                        "model_state": self.model.state_dict(),
                        "step": 0,
                        "committed_training_tokens": 0,
                        "architecture": self.architecture_description,
                    },
                    step=0,
                )
                report["checkpoint_roundtrip"] = "ok"
                report["checkpoint_path"] = str(info.path)
            except Exception as exc:
                report["checkpoint_roundtrip"] = f"failed: {exc}"

        try:
            import psutil

            process = psutil.Process()
            report["rss_bytes"] = int(process.memory_info().rss)
        except Exception:  # pragma: no cover
            report["rss_bytes"] = None

        report["parameters"] = int(self.model.num_parameters())
        self._emit("smoke_test", report)
        return report

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def request_stop(self) -> None:
        """Ask the training loop to stop after the current step."""

        self.stop_requested = True

    def train(self, max_steps: int | None = None) -> TrainingResult:
        """Run the training loop."""

        import torch

        if self.model is None:
            self.build()

        result = TrainingResult()
        started = time.perf_counter()
        self.model.train()

        step_limit = max_steps if max_steps is not None else self.training_config.get("max_steps", 0)
        try:
            for batch in self.iter_training_batches():
                if self.stop_requested:
                    self.stopper.stop_reason = StopReason.EXTERNAL_REQUEST
                    break
                if step_limit and self.step >= int(step_limit):
                    self.stopper.stop_reason = StopReason.STEP_LIMIT
                    break

                step_result = run_train_step(
                    model=self.model,
                    batch=batch,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    plan=self._plan(),
                    device=self.device,
                    grad_clip=self.grad_clip,
                    micro_batch_size=self.micro_batch_size,
                )
                self.step += 1
                self.metrics.step = self.step
                self.metrics.micro_steps += step_result.micro_batches
                self.metrics.add_loss(step_result.loss)
                self.metrics.add_tokens(step_result.tokens)
                self.committed_training_tokens += step_result.tokens

                if self.step % self.log_every == 0:
                    snapshot = self.metrics.snapshot(
                        learning_rate=step_result.learning_rate,
                        gradient_norm=step_result.gradient_norm,
                    )
                    self._emit("progress", snapshot.to_dict())

                if self.eval_every and self.step % self.eval_every == 0:
                    validation = self.run_validation()
                    self._emit("validation", validation.to_dict())

                if self.checkpoint_every and self.step % self.checkpoint_every == 0:
                    info = self.save_checkpoint()
                    if info is not None:
                        result.checkpoints.append(str(info.path))

                reason = self.stopper.evaluate(
                    tokens=self.committed_training_tokens, step=self.step
                )
                if reason:
                    self.stopper.stop_reason = reason
                    break
        except KeyboardInterrupt:  # pragma: no cover - interactive interruption
            self.stopper.stop_reason = StopReason.EXTERNAL_REQUEST
            self._emit("interrupted", {"step": self.step})
        except Exception as exc:
            self.stopper.stop_reason = StopReason.ERROR
            self._emit("error", {"message": str(exc)})
            raise

        elapsed = time.perf_counter() - started
        result.steps = self.step
        result.committed_training_tokens = self.committed_training_tokens
        result.validation_tokens = self.validation_tokens
        result.final_loss = self.metrics.mean_loss
        result.elapsed_seconds = elapsed
        result.tokens_per_second = self.committed_training_tokens / elapsed if elapsed > 0 else 0.0
        result.stop_reason = self.stopper.stop_reason or StopReason.COMPLETED

        final_info = self.save_checkpoint()
        if final_info is not None:
            result.checkpoints.append(str(final_info.path))

        self._emit("finished", result.to_dict())
        return result

    def _plan(self) -> Any:
        """Return the precision plan of this run.

        The plan carries the autocast configuration; the micro batch size
        and the number of accumulation steps are passed separately.
        """

        return self.precision

    def run_validation(self, limit: int = 20) -> Any:
        """Evaluate the model on the validation shards."""

        if not self.validation_shards:
            return None
        validation = evaluate_loss(
            self.model,
            self.iter_validation_batches(limit=limit),
            self.device,
            precision_plan=self.precision,
        )
        self.validation_tokens += validation.tokens
        return validation

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> Any:
        """Persist the full training state."""

        if self.checkpoint_manager is None:
            return None
        from ..checkpoint.manifest import CheckpointManifest

        payload = {
            "step": self.step,
            "committed_training_tokens": self.committed_training_tokens,
            "validation_tokens": self.validation_tokens,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "rng_state": capture_rng_state(self.seed).to_dict(),
            "dataset_position": self.dataset_position,
            "metrics": self.metrics.to_dict(),
            "recovery_state": self.recovery.to_dict() if self.recovery is not None else None,
            "architecture": self.architecture_description,
            "parameter_target": int(self.config.get("model", {}).get("parameter_target", 0)),
            "actual_parameters": int(self.architecture_description.get("actual_parameters", 0)),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "precision": self.precision.to_dict(),
        }
        manifest = CheckpointManifest(
            step=self.step,
            committed_training_tokens=self.committed_training_tokens,
            validation_tokens=self.validation_tokens,
            parameter_target=int(self.config.get("model", {}).get("parameter_target", 0)),
            actual_parameters=int(self.architecture_description.get("actual_parameters", 0)),
            context_length=CONTEXT_LENGTH,
            architecture=self.architecture_description,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            metrics=self.metrics.to_dict(),
        )
        info = self.checkpoint_manager.save(
            payload, step=self.step, manifest=manifest, sync=self.synchronizer
        )
        self._emit("checkpoint", {"path": str(info.path), "step": self.step})
        return info

    def restore_rng_state(self, state: dict[str, Any]) -> None:
        """Restore the RNG state captured in a checkpoint."""

        import random

        from ..utils.reproducibility import ReproducibilityState

        reproducible = ReproducibilityState.from_dict(state)
        if reproducible.python_state is not None:
            random.setstate(reproducible.python_state)
        if reproducible.numpy_state is not None:
            try:
                import numpy as np

                np.random.set_state(reproducible.numpy_state)
            except Exception:  # pragma: no cover
                pass
        if reproducible.torch_state is not None:
            try:
                import torch

                torch.set_rng_state(reproducible.torch_state)
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(TrainingEvent(kind=kind, payload=payload))
        except Exception:  # pragma: no cover - listeners must not break training
            pass

    def state_summary(self) -> dict[str, Any]:
        """Return a JSON friendly summary of the trainer state."""

        return {
            "step": self.step,
            "committed_training_tokens": self.committed_training_tokens,
            "validation_tokens": self.validation_tokens,
            "target_tokens": self.target_tokens,
            "device": self.device,
            "precision": self.precision.to_dict(),
            "metrics": self.metrics.to_dict(),
            "stop_reason": self.stopper.stop_reason,
        }
