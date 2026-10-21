"""The inference engine.

The engine owns the artifact, the backend, the tokenizer, the budget
pipeline and the metrics.  The API layer only translates requests; all
execution happens here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from ..artifacts.loader import LoadedArtifact, load_artifact
from ..artifacts.registry import ArtifactRegistry
from ..backends.registry import BackendRegistry, BackendSelection
from ..budget.budget_allocator import BudgetAllocator, ResourceState
from ..budget.complexity_estimator import ComplexityEstimator
from ..budget.controller import BaselineBudgetController, BudgetLimits
from ..budget.safeguards import SafeguardPolicy, Safeguards
from ..budget.scheduler import BudgetScheduler
from ..budget.telemetry import BudgetTelemetry
from ..cache.cache_policy import CachePolicy
from ..cache.prefix_cache import PrefixCache
from ..cache.token_cache import TokenCache
from ..generation.generator import GenerationRequest, Generator, TokenChunk
from ..orchestration.request_pipeline import InferenceRequest, PipelineResult, RequestPipeline
from ..scheduling.admission import AdmissionController
from ..scheduling.cancellation import CancellationRegistry
from ..scheduling.continuous_batching import ContinuousBatchingScheduler
from ..utils.errors import ArtifactError, RequestError
from .memory import MemoryAccountant

__all__ = ["EngineConfig", "InferenceEngine"]


@dataclass
class EngineConfig:
    """The configuration of the inference engine."""

    model_paths: list[str] = field(default_factory=list)
    default_model_alias: str = "tinycode-local"
    backend: str = "auto"
    threads: int | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    api_keys: list[str] = field(default_factory=list)
    require_api_key: bool = True
    max_active_requests: int = 8
    queue_capacity: int = 64
    request_body_limit_bytes: int = 1 << 20
    rate_limit_per_minute: int = 0
    budget: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    telemetry_log: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EngineConfig":
        """Build a configuration from a dictionary."""

        payload = dict(payload or {})
        return cls(
            model_paths=[str(path) for path in payload.get("model_paths", []) or []],
            default_model_alias=str(payload.get("default_model_alias", "tinycode-local")),
            backend=str(payload.get("backend", "auto")),
            threads=payload.get("threads"),
            host=str(payload.get("host", "127.0.0.1")),
            port=int(payload.get("port", 8000)),
            api_keys=[str(key) for key in payload.get("api_keys", []) or []],
            require_api_key=bool(payload.get("require_api_key", True)),
            max_active_requests=int(payload.get("max_active_requests", 8)),
            queue_capacity=int(payload.get("queue_capacity", 64)),
            request_body_limit_bytes=int(payload.get("request_body_limit_bytes", 1 << 20)),
            rate_limit_per_minute=int(payload.get("rate_limit_per_minute", 0)),
            budget=dict(payload.get("budget", {}) or {}),
            cache=dict(payload.get("cache", {}) or {}),
            telemetry_log=payload.get("telemetry_log"),
        )


class InferenceEngine:
    """Serves the loaded TinyCode artifacts."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.registry = ArtifactRegistry()
        self.memory = MemoryAccountant()
        self.telemetry = BudgetTelemetry(log_path=self.config.telemetry_log)
        self.cancellations = CancellationRegistry()
        self.backend_selection: BackendSelection | None = None
        self.backend: Any = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.generator: Generator | None = None
        self.pipeline: RequestPipeline | None = None
        self.scheduler: ContinuousBatchingScheduler | None = None
        self.token_cache = TokenCache(CachePolicy().validate())
        self.cache_policy = CachePolicy().validate()
        self.prefix_cache = PrefixCache(self.cache_policy)
        self.started_at: float | None = None
        self.load_seconds: float = 0.0
        self._lock = threading.RLock()
        self._request_counter = 0
        self.metrics: dict[str, float] = {
            "requests_received": 0,
            "requests_completed": 0,
            "requests_rejected": 0,
            "requests_cancelled": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "first_token_ms_total": 0.0,
            "response_ms_total": 0.0,
        }

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def load_artifact(self, directory: str, alias: str | None = None) -> LoadedArtifact:
        """Load an artifact and prepare the runtime for it."""

        started = time.perf_counter()
        artifact = self.registry.load(directory, alias=alias)
        if self.backend is None:
            ir_path = artifact.openvino_model_path
            self.backend_selection = BackendRegistry(
                {
                    "preferred": self.config.backend,
                    "threads": self.config.threads,
                    "openvino_model": str(ir_path) if ir_path else None,
                }
            ).select()
            self.backend = self.backend_selection.backend
        if self.backend is None:
            raise ArtifactError(
                "No usable backend was found: "
                + (self.backend_selection.reason if self.backend_selection else "unknown")
            )

        self.model = artifact.model
        self.tokenizer = artifact.tokenizer
        self.generator = Generator(
            self.model, self.tokenizer, backend=self._generation_backend()
        )

        limits = BudgetLimits(**{
            key: value
            for key, value in (self.config.budget or {}).items()
            if key in BudgetLimits.__dataclass_fields__
        })
        controller = BaselineBudgetController()
        trained = self._load_trained_controller(artifact)
        if trained is not None:
            controller = trained

        allocator = BudgetAllocator(controller=controller, limits=limits)
        safeguards = Safeguards(
            SafeguardPolicy(limits=limits, max_request_bytes=self.config.request_body_limit_bytes)
        )
        admission = AdmissionController(
            memory=self.memory,
            max_active_requests=self.config.max_active_requests,
        )
        self.pipeline = RequestPipeline(
            generator=self.generator,
            allocator=allocator,
            complexity=ComplexityEstimator(),
            scheduler=BudgetScheduler(),
            safeguards=safeguards,
            telemetry=self.telemetry,
            resource_provider=self.resource_state,
            controller_description=controller.describe,
            prefix_finder=self.find_cached_prefix,
        )
        self.scheduler = ContinuousBatchingScheduler(
            session_factory=self._create_session,
            max_active=self.config.max_active_requests,
            queue_capacity=self.config.queue_capacity,
            admission=admission,
            cancellations=self.cancellations,
        )
        self.load_seconds = time.perf_counter() - started
        self.started_at = self.started_at or time.time()
        return artifact

    def _generation_backend(self) -> Any:
        """Return the backend that executes the generation.

        The CPU backend wraps the numpy model, so it is not used as the
        generation backend (the model is called directly).  An accelerator
        backend (CUDA, Intel GPU) is used, because it owns its own
        execution path.
        """

        if self.backend is None:
            return None
        if self.backend.name == "cpu":
            return None
        return self.backend

    def _load_trained_controller(self, artifact: LoadedArtifact) -> Any:
        controller_info = artifact.controller
        if not controller_info:
            return None
        state_dict = controller_info.get("state_dict")
        if state_dict is None:
            return None
        from ..budget.controller import TrainedBudgetController

        try:
            controller = TrainedBudgetController(state_dict, controller_info.get("metadata"))
        except Exception:
            return None
        if not controller.compatible:
            return None
        return controller

    def _create_session(self, request_id: str, payload: Any) -> Any:
        if self.generator is None:
            raise RuntimeError("No model has been loaded.")
        request: GenerationRequest = payload
        return self.generator.new_session(request)

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def next_request_id(self) -> str:
        """Return a unique request id."""

        with self._lock:
            self._request_counter += 1
            return f"chatcmpl-{int(time.time())}-{self._request_counter:06d}"

    def resource_state(self) -> ResourceState:
        """Return the current resource state of the runtime."""

        queue_depth = self.scheduler.queue.stats()["depth"] if self.scheduler else 0
        active = len(self.scheduler.active) if self.scheduler else 0
        snapshot = self.memory.snapshot()
        limit = self.config.max_active_requests or 1
        return ResourceState(
            queue_depth=queue_depth,
            queue_capacity=self.config.queue_capacity,
            memory_pressure=0.0,
            backend_tokens_per_second=self._estimate_speed(),
            available_memory_bytes=int(snapshot.reserved_bytes),
            active_requests=active,
        )

    def _estimate_speed(self) -> float:
        """A rough tokens/second estimate used for the latency estimate."""

        if self.backend is None:
            return 32.0
        accelerated = bool(self.backend.capabilities().is_accelerated)
        return 256.0 if accelerated else 48.0

    def tokenize(self, text: str) -> list[int]:
        """Tokenise a text with the tokenizer of the loaded model."""

        if self.tokenizer is None:
            raise RequestError("No tokenizer is loaded.")
        return self.tokenizer.encode(text)

    # ------------------------------------------------------------------
    # caching
    # ------------------------------------------------------------------
    def _cache_identity(self, model_id: str | None = None) -> tuple[str, str, str]:
        """Return the ``(model_id, model_version, fingerprint)`` triple."""

        available = self.registry.model_ids()
        if not available:
            raise RequestError("No model is loaded into the engine.")
        model_id = model_id or available[0]
        if model_id not in available:
            raise RequestError(f"The model '{model_id}' is not loaded.")
        artifact = self.registry.get(model_id)
        return (
            model_id,
            artifact.manifest.model_version,
            artifact.manifest.tokenizer_fingerprint,
        )

    def cache_input(
        self,
        text: str,
        model_id: str | None = None,
        compute_prefix: bool = True,
    ) -> dict[str, Any]:
        """Cache an input so that later requests can reuse it.

        The tokenised input is stored in the token cache and, when
        ``compute_prefix`` is set, the KV state of the input is computed
        once and stored in the prefix cache.  A later request whose prompt
        starts with the cached input only computes the remaining tokens.

        Args:
            text: the input to cache.
            model_id: the model the input belongs to.
            compute_prefix: also compute and store the KV state.

        Returns:
            A summary of what was cached.
        """

        if self.tokenizer is None or self.model is None:
            raise RequestError("No model is loaded into the engine.")
        if not text or not text.strip():
            raise RequestError("The input to cache must not be empty.")

        token_ids = self.tokenizer.encode(text)
        if len(token_ids) > 1024:
            token_ids = token_ids[-1024:]
        if not token_ids:
            raise RequestError("The input to cache did not produce any token.")

        model_id, model_version, fingerprint = self._cache_identity(model_id)
        self.token_cache.put(model_id, model_version, fingerprint, text, token_ids)

        state_stored = False
        compute_ms = 0.0
        if compute_prefix:
            import time

            import numpy as np

            from ..cache.kv_cache import KVCache

            cache = KVCache(
                n_layers=int(self.model.config["n_layers"]),
                n_heads=int(self.model.config["n_heads"]),
                head_dim=int(self.model.config["d_model"]) // int(self.model.config["n_heads"]),
                context_length=int(self.model.capabilities().context_length),
            )
            started = time.perf_counter()
            try:
                logits = self.model.forward(np.asarray(token_ids, dtype=np.int64), cache=cache)
                payload = {
                    "length": cache.length,
                    "logits": np.asarray(logits, dtype=np.float32).copy(),
                    "keys": [cache.keys[layer][:, : cache.length, :].copy() for layer in range(cache.n_layers)],
                    "values": [
                        cache.values[layer][:, : cache.length, :].copy() for layer in range(cache.n_layers)
                    ],
                }
            finally:
                compute_ms = (time.perf_counter() - started) * 1000.0
                cache.release()
            self.prefix_cache.put(
                model_id, model_version, fingerprint, token_ids, payload=payload
            )
            state_stored = True

        return {
            "model": model_id,
            "tokens": len(token_ids),
            "token_ids": token_ids,
            "prefix_state_stored": state_stored,
            "compute_ms": round(compute_ms, 3),
            "text": text,
        }

    def find_cached_prefix(self, request: InferenceRequest) -> Any:
        """Return the cached prefix of a request, if there is one."""

        if self.tokenizer is None:
            return None
        model_id, model_version, fingerprint = self._cache_identity(request.model_id or None)
        return self.prefix_cache.find_prefix(
            model_id, model_version, fingerprint, list(request.prompt_ids)
        )

    def clear_cache(self) -> dict[str, Any]:
        """Clear the token and the prefix cache."""

        self.token_cache.clear()
        self.prefix_cache.clear()
        return self.cache_stats()

    def cache_stats(self) -> dict[str, Any]:
        """Return the statistics of the caches."""

        return {
            "token_cache": self.token_cache.stats(),
            "prefix_cache": self.prefix_cache.stats(),
            "policy": self.cache_policy.to_dict(),
        }

    def generate(
        self,
        request: InferenceRequest,
        cancelled: Callable[[], bool] | None = None,
        on_token: Callable[[TokenChunk], None] | None = None,
    ) -> PipelineResult:
        """Run a request through the pipeline."""

        if self.pipeline is None:
            raise RequestError("No model has been loaded into the engine.")
        self.metrics["requests_received"] += 1
        started = time.perf_counter()
        try:
            result = self.pipeline.handle(request, cancelled=cancelled, on_token=on_token)
        except Exception:
            self.metrics["requests_rejected"] += 1
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        self.metrics["requests_completed"] += 1
        self.metrics["input_tokens"] += result.prompt_tokens
        self.metrics["output_tokens"] += result.completion_tokens
        self.metrics["response_ms_total"] += elapsed
        if result.first_token_ms is not None:
            self.metrics["first_token_ms_total"] += result.first_token_ms
        return result

    def stream(
        self,
        request: InferenceRequest,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[TokenChunk]:
        """Stream a request.

        The streaming path uses the same budget pipeline as the blocking
        path; the tokens are emitted while the answer is generated.
        """

        if self.pipeline is None:
            raise RequestError("No model has been loaded into the engine.")

        import queue
        import threading

        chunks: "queue.Queue[Any]" = queue.Queue()
        sentinel = object()

        def _on_token(chunk: TokenChunk) -> None:
            chunks.put(chunk)

        def _worker() -> None:
            try:
                self.generate(request, cancelled=cancelled, on_token=_on_token)
            except Exception as exc:  # pragma: no cover - surfaced to the caller
                chunks.put(exc)
            finally:
                chunks.put(sentinel)

        self.metrics["requests_received"] += 1
        thread = threading.Thread(target=_worker, name="tinycode-stream", daemon=True)
        thread.start()
        try:
            while True:
                item = chunks.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def model_ids(self) -> list[str]:
        return self.registry.model_ids()

    def describe(self) -> dict[str, Any]:
        """Return a description of the engine state."""

        return {
            "models": self.registry.model_ids(),
            "backend": self.backend_selection.to_dict() if self.backend_selection else None,
            "loaded": bool(self.model is not None),
            "uptime_seconds": round(time.time() - self.started_at, 3) if self.started_at else 0.0,
            "load_seconds": round(self.load_seconds, 3),
            "context_length": 1024,
            "metrics": dict(self.metrics),
        }

    def shutdown(self) -> None:
        """Release the resources of the engine."""

        if self.backend is not None:
            try:
                self.backend.shutdown()
            except Exception:  # pragma: no cover
                pass
            self.backend = None
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.pipeline = None
        self.scheduler = None
        self.started_at = None
