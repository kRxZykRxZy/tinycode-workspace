"""Interactive workflows of the model builder."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..budget.controller import BaselineBudgetController, TrainableBudgetController
from ..budget.features import BudgetFeatures
from ..checkpoint.manager import CheckpointManager
from ..checkpoint.retention import RetentionPolicy
from ..configuration.parameter_parser import format_parameter_target
from ..data.access_validation import validate_dataset_access
from ..data.pipeline import prepare_dataset
from ..evaluation.code_tasks import DEFAULT_CODE_TASKS
from ..evaluation.benchmark_runner import BenchmarkRunner
from ..export.pipeline import export_artifact
from ..export.verification import verify_artifact
from ..hardware.capability_report import build_capability_report
from ..model.architecture import TransformerConfig
from ..model.factory import NoExactArchitectureError, solve_architecture
from ..model.parameter_accounting import analytic_parameter_count, parameter_breakdown
from ..storage.storage_policy import StoragePolicy, resolve_storage
from ..tokenizer.trainer import train_tokenizer, train_tokenizer_from_texts
from ..utils.paths import ensure_dir
from .prompts import ask_choice, ask_yes_no, confirm

__all__ = [
    "workflow_hardware",
    "workflow_configure",
    "workflow_tokenizer",
    "workflow_data",
    "workflow_train",
    "workflow_evaluate",
    "workflow_export",
    "workflow_controller",
    "workflow_dashboard",
    "workflow_diagnostics",
]


def _header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


# --------------------------------------------------------------------------
def workflow_hardware(session: Any) -> None:
    """Show the capabilities of the current machine."""

    _header("Hardware capabilities")
    report = build_capability_report()
    session.hardware = report.profile
    print(report.to_text())
    if report.profile.is_colab:
        print("\n  Google Colab detected: checkpoints can be mirrored to Google Drive.")


# --------------------------------------------------------------------------
def workflow_configure(session: Any) -> None:
    """Select the parameter target and solve the architecture."""

    _header("Model configuration")
    from ..application.prompts import ask_parameter_target

    target = ask_parameter_target(default=session.state.parameter_target)
    session.set_parameter_target(target)
    print(f"\n  Selected parameter target: {target:,} ({format_parameter_target(target)})")

    model_cfg = session.config.get("model", {}) or {}
    vocab_preference = list(model_cfg.get("vocab_preference", [98_304]))
    try:
        solution = solve_architecture(
            target=target,
            vocab_preference=vocab_preference,
            preferred_vocab=int(model_cfg.get("vocab_size", vocab_preference[0])),
            max_layers=int(model_cfg.get("max_layers", 64)),
            preferred_ff_ratio=float(model_cfg.get("preferred_ff_ratio", 4.0)),
            min_head_dim=int(model_cfg.get("min_head_dim", 16)),
        )
    except NoExactArchitectureError as exc:
        print(f"\n  {exc}")
        print("  Suggested alternatives:")
        for value in (1_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000):
            print(f"    - {format_parameter_target(value)}")
        return

    config = solution.config
    breakdown = parameter_breakdown(config)
    print(f"\n  Architecture solved:")
    print(f"    d_model        : {config.d_model}")
    print(f"    layers         : {config.n_layers}")
    print(f"    heads          : {config.n_heads} (head dim {config.head_dim})")
    print(f"    d_ff           : {config.d_ff} (ratio {config.ff_ratio:.2f})")
    print(f"    vocabulary     : {config.vocab_size:,}")
    print(f"    norm           : {config.norm_type}")
    print(f"    positions      : {config.position_encoding}")
    print(f"    tied embeddings: {config.tie_embeddings}")
    print(f"    context length : {config.context_length} (fixed)")
    print(f"    parameters     : {breakdown.total:,} (target {target:,})")
    if solution.warnings:
        for warning in solution.warnings:
            print(f"    ! {warning}")

    if breakdown.total != target:  # pragma: no cover - guarded by the solver
        print("    ! parameter mismatch; refusing to continue")
        return

    session.state.architecture = config.to_dict()
    session.state.actual_parameters = breakdown.total
    session.state.vocabulary_size = config.vocab_size
    session.update_config("model", {"vocab_size": config.vocab_size})
    session.save()
    print("\n  Configuration stored in the session.")


# --------------------------------------------------------------------------
def workflow_tokenizer(session: Any) -> None:
    """Train the coding tokenizer."""

    _header("Tokenizer training")
    from ..application.prompts import ask_int

    tokenizer_cfg = session.config.get("tokenizer", {}) or {}
    vocab_size = int(tokenizer_cfg.get("vocab_size", 98_304))
    if session.state.vocabulary_size:
        vocab_size = min(vocab_size, max(1024, session.state.vocabulary_size))

    print(f"  Target vocabulary size: {vocab_size:,}")
    use_local = ask_yes_no("Train from a local directory of source files?", default=False)
    output_dir = ensure_dir(session.paths.tokenizer_dir / "current")

    if use_local:
        from ..application.prompts import ask_text

        corpus_dir = ask_text("  Directory with the training corpus")
        directory = Path(corpus_dir)
        if not directory.is_dir():
            print("  Directory not found; falling back to a synthetic smoke corpus.")
            use_local = False

    if use_local:
        files = [
            str(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".txt"}
        ][:5000]
        if not files:
            print("  No usable files found; falling back to a synthetic smoke corpus.")
            use_local = False
        else:
            print(f"  Training on {len(files)} files ...")
            result = train_tokenizer(files, output_dir, vocab_size=vocab_size)
            session.state.tokenizer_dir = str(output_dir)
            session.state.tokenizer_fingerprint = result.fingerprint
            session.save()
            print(f"  Tokenizer trained: vocab {result.vocab_size:,}, fingerprint {result.fingerprint}")
            return

    print("  Training a small synthetic tokenizer (smoke corpus) ...")
    samples = [
        "def add(a, b):\n    return a + b\n\n" for _ in range(200)
    ] + [
        "function multiply(a, b) { return a * b; }\n" for _ in range(200)
    ]
    result = train_tokenizer_from_texts(samples, output_dir, vocab_size=min(vocab_size, 4096))
    session.state.tokenizer_dir = str(output_dir)
    session.state.tokenizer_fingerprint = result.fingerprint
    session.save()
    print(f"  Tokenizer trained: vocab {result.vocab_size:,}, fingerprint {result.fingerprint}")
    print("  Note: this smoke tokenizer is not suitable for real training.")


# --------------------------------------------------------------------------
def workflow_data(session: Any) -> None:
    """Prepare the dataset from The Stack."""

    _header("Dataset preparation")
    from ..application.prompts import ask_text

    dataset_cfg = session.config.get("dataset", {}) or {}
    output_dir = ensure_dir(session.paths.data_dir / "prepared")

    print("  Validating dataset access ...")
    report = validate_dataset_access(
        repo_id=dataset_cfg.get("repo_id"), subset=dataset_cfg.get("subset")
    )
    print(f"  {report.reason}")
    if not report.ok:
        print("  Continuing anyway; the download step will report any problem.")

    source = "the-stack"
    local_corpus = None
    if ask_yes_no("  Use a local corpus directory instead of The Stack?", default=False):
        local_corpus = ask_text("  Corpus directory")
        source = "local"

    tokenizer = None
    eos_token_id = 0
    if session.state.tokenizer_dir:
        from ..tokenizer.loader import load_tokenizer

        tokenizer = load_tokenizer(Path(session.state.tokenizer_dir) / "tokenizer.json")
        eos_token_id = tokenizer.token_to_id("<|endoftext|>") or 0
        print(f"  Using tokenizer {session.state.tokenizer_fingerprint}")

    result = prepare_dataset(
        session.config,
        output_dir,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        source=source,
        local_corpus=local_corpus,
    )

    session.state.dataset_dir = str(output_dir)
    session.state.train_shards = [str(entry["path"]) for entry in result.train_shards]
    session.state.validation_shards = [str(entry["path"]) for entry in result.validation_shards]
    session.state.dataset_tokens = result.training_tokens
    session.save()

    print(f"\n  Prepared {len(result.train_shards)} train shards and "
          f"{len(result.validation_shards)} validation shards")
    print(f"  Training tokens: {result.training_tokens:,}")
    print(f"  Validation tokens: {result.validation_tokens:,} (never counted towards the target)")


# --------------------------------------------------------------------------
def workflow_train(session: Any) -> None:
    """Train the model."""

    _header("Training")
    from ..application.prompts import ask_int
    from ..training.trainer import Trainer

    if not session.state.architecture:
        print("  Configure the model first (menu option 2).")
        return
    if not session.state.train_shards:
        print("  Prepare the dataset first (menu option 3).")
        return

    config = TransformerConfig.from_dict(session.state.architecture)
    run_dir = ensure_dir(session.paths.runs_dir / f"run-{int(time.time())}")
    checkpoint_dir = ensure_dir(session.paths.checkpoints_dir / run_dir.name)

    manager = CheckpointManager(
        checkpoint_dir,
        retention=RetentionPolicy(
            keep_last=int(session.config.get("checkpoint", {}).get("keep_last", 3)),
            keep_every=int(session.config.get("checkpoint", {}).get("keep_every", 10)),
        ),
    )

    storage, description = resolve_storage(
        StoragePolicy(local_root=session.paths.root, drive_folder="tinycode")
    )
    print(f"  Checkpoint storage: {description}")

    trainer = Trainer(
        config=session.config,
        model_config=config,
        run_dir=run_dir,
        train_shards=session.state.train_shards,
        validation_shards=session.state.validation_shards,
        tokenizer_fingerprint=session.state.tokenizer_fingerprint or "",
        checkpoint_manager=manager,
    )

    print("\n  Running the smoke test ...")
    smoke = trainer.smoke_test()
    print(f"    forward loss      : {smoke.get('forward_loss')}")
    print(f"    gradient norm     : {smoke.get('gradient_norm')}")
    print(f"    optimizer step    : {smoke.get('optimizer_step')}")
    print(f"    checkpoint write  : {smoke.get('checkpoint_roundtrip')}")
    print(f"    parameters        : {smoke.get('parameters'):,}" if smoke.get("parameters") else "")

    tokens = session.state.training_token_target
    print(f"\n  Target: {tokens:,} training tokens")
    print(f"  Effective batch: {session.config['training']['batch_size']} sequences "
          f"x 1024 tokens")
    if not confirm("  Start the training run now?", default=True):
        print("  Training cancelled.")
        return

    trainer.build()
    result = trainer.train()
    session.state.last_training = result.to_dict()
    session.state.checkpoint_dir = str(checkpoint_dir)
    session.save()

    print(f"\n  Training finished: {result.stop_reason}")
    print(f"    steps                  : {result.steps:,}")
    print(f"    committed tokens       : {result.committed_training_tokens:,}")
    print(f"    final training loss    : {result.final_loss:.4f}")
    print(f"    tokens per second      : {result.tokens_per_second:.1f}")
    print(f"    checkpoints            : {len(result.checkpoints)}")


# --------------------------------------------------------------------------
def workflow_evaluate(session: Any) -> None:
    """Evaluate the trained model."""

    _header("Evaluation")
    from ..model.causal_model import build_model
    from ..tokenizer.loader import load_tokenizer

    if not session.state.architecture:
        print("  Configure the model first (menu option 2).")
        return

    config = TransformerConfig.from_dict(session.state.architecture)
    model = build_model(config)

    checkpoint = _latest_checkpoint(session)
    if checkpoint is not None:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model_state", payload))
        print(f"  Loaded weights from {checkpoint}")
    else:
        print("  No checkpoint found; evaluating the untrained model.")

    tokenizer = None
    if session.state.tokenizer_dir:
        tokenizer = load_tokenizer(Path(session.state.tokenizer_dir) / "tokenizer.json")

    runner = BenchmarkRunner(model=model, tokenizer=tokenizer)

    def _generate(task: Any) -> str:
        if tokenizer is None:
            return ""
        import torch

        from ..configuration.immutable_constraints import CONTEXT_LENGTH

        prompt_ids = list(tokenizer.encode(task.prompt).ids)
        prompt_ids = prompt_ids[-(CONTEXT_LENGTH - 1) :]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            for _ in range(min(task.max_tokens, 128)):
                logits = model(input_ids).logits[:, -1, :]
                next_token = int(torch.argmax(logits, dim=-1).item())
                if next_token <= 0:
                    break
                prompt_ids.append(next_token)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1
                )
                if input_ids.shape[1] >= CONTEXT_LENGTH:
                    break
        return tokenizer.decode(prompt_ids[len(prompt_ids) - min(task.max_tokens, 128) :])

    report = runner.run(batches=None, generate=_generate if tokenizer is not None else None)
    report.add("model", {"parameters": model.num_parameters(), "architecture": config.to_dict()})
    output = ensure_dir(session.paths.reports_dir / "latest")
    report.write(output / "evaluation.json")
    print(f"  Report written to {output / 'evaluation.json'}")


def _latest_checkpoint(session: Any) -> str | None:
    directory = session.state.checkpoint_dir
    if not directory:
        return None
    candidates = sorted(Path(directory).glob("checkpoint-*.pt"))
    return str(candidates[-1]) if candidates else None


# --------------------------------------------------------------------------
def workflow_export(session: Any) -> None:
    """Export the artifact for the inference engine."""

    _header("Artifact export")
    from ..model.causal_model import build_model

    if not session.state.architecture:
        print("  Configure the model first (menu option 2).")
        return

    config = TransformerConfig.from_dict(session.state.architecture)
    model = build_model(config)

    checkpoint = _latest_checkpoint(session)
    if checkpoint is not None:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model_state", payload))
        print(f"  Loaded weights from {checkpoint}")

    output_dir = ensure_dir(session.paths.artifacts_dir / "tinycode-latest")
    export_cfg = session.config.get("export", {}) or {}
    wants_openvino = bool(export_cfg.get("openvino", False))
    if ask_yes_no("  Also export an OpenVINO IR (for the Intel GPU path)?", default=wants_openvino):
        wants_openvino = True

    manifest = export_artifact(
        model,
        output_dir,
        tokenizer_dir=session.state.tokenizer_dir,
        model_name=session.config.get("project", {}).get("name", "tinycode"),
        model_version=session.config.get("project", {}).get("version", "0.1.0"),
        parameter_target=session.state.parameter_target,
        controller_dir=session.state.controller_dir,
        training_metadata=session.state.last_training,
        openvino=wants_openvino,
        openvino_device=str(export_cfg.get("openvino_device", "GPU")),
    )
    session.state.artifact_dir = str(output_dir)
    session.save()

    print(f"  Artifact written to {output_dir}")
    print(f"    format version : {manifest['artifact_format_version']}")
    print(f"    parameters     : {manifest['parameters']['actual_trainable']:,}")
    print(f"    context length : {manifest['architecture']['context_length']}")
    if manifest["export"].get("openvino", {}).get("present"):
        print(f"    OpenVINO IR    : {manifest['export']['openvino']['file']} (verified)")

    report = verify_artifact(output_dir)
    print(f"  Verification: {'OK' if report.ok else 'FAILED'}")
    for problem in report.problems:
        print(f"    - {problem}")


# --------------------------------------------------------------------------
def workflow_controller(session: Any) -> None:
    """Train the optional adaptive budget controller."""

    _header("Adaptive budget controller")
    from ..budget.policy_dataset import build_policy_dataset
    from ..budget.train_controller import train_controller

    controller_cfg = session.config.get("budget_controller", {}) or {}
    records_path = session.paths.runs_dir / "policy_records.json"
    if not records_path.is_file():
        print("  No policy records found.")
        print("  The controller can only be trained on recorded request outcomes.")
        print("  Until then the inference engine uses the deterministic baseline policy.")
        return

    records = json.loads(records_path.read_text(encoding="utf-8"))
    dataset = build_policy_dataset(records)
    print(f"  Loaded {len(dataset)} policy records.")

    baseline = BaselineBudgetController()
    decision = baseline.decide(
        BudgetFeatures(complexity_score=0.5, requested_output_tokens=256), 256
    )
    print(f"  Baseline decision for a mid complexity request: {decision.to_dict()}")

    output_dir = ensure_dir(session.paths.artifacts_dir / "controller")
    result = train_controller(
        dataset,
        hidden_sizes=list(controller_cfg.get("hidden_sizes", [64, 32])),
        epochs=int(controller_cfg.get("epochs", 50)),
        learning_rate=float(controller_cfg.get("learning_rate", 1e-3)),
        output_dir=output_dir,
    )
    session.state.controller_dir = str(output_dir)
    session.save()

    print(f"\n  Controller trained on {result.samples} samples")
    print(f"    train loss      : {result.train_loss:.6f}")
    print(f"    validation loss : {result.validation_loss:.6f}")
    print(f"    artifact        : {output_dir}")
    print("  The controller is optional; the engine falls back to the baseline policy.")


# --------------------------------------------------------------------------
def workflow_dashboard(session: Any) -> None:
    """Start the authenticated dashboard."""

    _header("Training dashboard")
    from ..dashboard.authentication import DashboardAuth, read_or_create_token
    from ..dashboard.controls import ControlBridge
    from ..dashboard.server import DashboardServer, create_dashboard_app
    from ..dashboard.state import DashboardState
    from ..dashboard.tunnel import CloudflareTunnel, is_cloudflared_available

    dashboard_cfg = session.config.get("dashboard", {}) or {}
    host = str(dashboard_cfg.get("host", "127.0.0.1"))
    port = int(dashboard_cfg.get("port", 8765))

    token_path = session.paths.runs_dir / "dashboard_token.txt"
    token = dashboard_cfg.get("auth_token") or read_or_create_token(token_path)
    auth = DashboardAuth(token, required=True)
    state = DashboardState()
    controls = ControlBridge(state)
    app = create_dashboard_app(state=state, controls=controls, auth=auth)
    server = DashboardServer(app, host=host, port=port)
    server.start_background()

    url = f"http://{host}:{port}/?token={token}"
    print(f"  Dashboard available at: {url}")
    print(f"  (the token is also stored in {token_path})")

    if dashboard_cfg.get("cloudflare_tunnel"):
        if is_cloudflared_available():
            handle = CloudflareTunnel(port=port, host=host).start()
            if handle.url:
                print(f"  Cloudflare quick tunnel: {handle.url}")
                print("  The tunnel URL is not a substitute for the dashboard token.")
            else:
                print(f"  Cloudflare tunnel not started: {handle.error}")
        else:
            print("  cloudflared is not installed; the tunnel was skipped.")

    input("  Press Enter to stop the dashboard and return to the menu...")


# --------------------------------------------------------------------------
def workflow_diagnostics(session: Any) -> None:
    """Print the environment diagnostics."""

    _header("Environment diagnostics")
    from ..telemetry.diagnostics import collect_diagnostics

    print(collect_diagnostics().to_text())
