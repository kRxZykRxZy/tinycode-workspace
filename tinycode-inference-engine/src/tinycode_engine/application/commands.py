"""The commands of the inference engine CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..observability.diagnostics import collect_diagnostics
from ..runtime.engine import EngineConfig, InferenceEngine
from ..server.startup import start_engine
from .service import EngineService, ServiceConfig, load_config

__all__ = ["command_serve", "command_doctor", "command_models", "command_tokenize", "command_bench"]


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def command_serve(
    model: str | None = None,
    host: str | None = None,
    port: int | None = None,
    config_path: str | None = None,
    api_key: str | None = None,
    workers: int = 1,
) -> int:
    """Start the OpenAI compatible server."""

    config = load_config(config_path)
    if model:
        config.model_paths = [model]
    if host:
        config.host = host
    if port:
        config.port = port
    if api_key:
        config.api_keys = [api_key]
        config.require_api_key = True

    service = EngineService(
        config=config,
        service_config=ServiceConfig(host=config.host, port=config.port, workers=workers),
    )
    service.build()
    report = service.start_models()
    print(f"tinycode-inference-engine")
    print(f"  endpoint   : http://{config.host}:{config.port}/v1")
    print(f"  models     : {', '.join(report.models) or 'none'}")
    print(f"  backend    : {report.backend} ({report.backend_reason})")
    print(f"  context    : 1024 tokens (fixed)")
    if report.errors:
        for error in report.errors:
            print(f"  error      : {error}")
    if not report.models:
        print("  no model was loaded; the server will answer with an error")
    try:
        service.run()
    except KeyboardInterrupt:  # pragma: no cover
        print("\nshutting down")
    finally:
        service.shutdown()
    return 0


def command_doctor(config_path: str | None = None) -> int:
    """Print the diagnostics of the runtime environment."""

    diagnostics = collect_diagnostics()
    print(diagnostics.to_text())

    config = load_config(config_path)
    engine = InferenceEngine(config)
    report = start_engine(config, engine)
    print("\nstartup report:")
    _print(report.to_dict())
    return 0 if report.ok else 1


def command_models(config_path: str | None = None) -> int:
    """List the models that can be loaded."""

    config = load_config(config_path)
    payload = {
        "model_paths": config.model_paths,
        "default_model_alias": config.default_model_alias,
        "existing": [path for path in config.model_paths if Path(path).is_dir()],
    }
    _print(payload)
    return 0


def command_tokenize(text: str, model: str | None = None, config_path: str | None = None) -> int:
    """Tokenise a text with the tokenizer of a model."""

    config = load_config(config_path)
    if model:
        config.model_paths = [model]
    engine = InferenceEngine(config)
    report = start_engine(config, engine)
    if not report.models:
        print("no model could be loaded", file=sys.stderr)
        return 1
    token_ids = engine.tokenize(text)
    _print(
        {
            "text": text,
            "tokens": len(token_ids),
            "ids": token_ids,
            "decoded": engine.tokenizer.decode(token_ids) if engine.tokenizer else None,
        }
    )
    return 0


def command_bench(
    model: str | None = None,
    prompt: str = "Write a Python function that adds two numbers.",
    max_tokens: int = 64,
    iterations: int = 1,
    config_path: str | None = None,
) -> int:
    """Run a small local benchmark against a model."""

    import time

    from ..orchestration.request_pipeline import InferenceRequest

    config = load_config(config_path)
    if model:
        config.model_paths = [model]
    engine = InferenceEngine(config)
    report = start_engine(config, engine)
    if not report.models:
        print("no model could be loaded", file=sys.stderr)
        return 1

    latencies: list[float] = []
    tokens = 0
    for index in range(max(1, iterations)):
        request = InferenceRequest(
            request_id=f"bench-{index}",
            model_id=report.models[0],
            prompt=prompt,
            prompt_ids=engine.tokenize(prompt),
            max_tokens=max_tokens,
            temperature=0.0,
        )
        started = time.perf_counter()
        result = engine.generate(request)
        latencies.append((time.perf_counter() - started) * 1000.0)
        tokens += result.completion_tokens

    total_seconds = sum(latencies) / 1000.0
    _print(
        {
            "iterations": len(latencies),
            "latency_ms_mean": round(sum(latencies) / len(latencies), 3),
            "latency_ms_min": round(min(latencies), 3),
            "tokens": tokens,
            "tokens_per_second": round(tokens / total_seconds, 2) if total_seconds else 0.0,
        }
    )
    return 0
