"""The end to end request pipeline.

The pipeline implements the ten stages of the adaptive budget design:

    intake -> complexity -> allocation -> plan -> execution ->
    uncertainty -> decision -> verification -> finalisation -> telemetry
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..budget.budget_allocator import BudgetAllocator, ResourceState
from ..budget.complexity_estimator import ComplexityEstimator
from ..budget.confidence import ConfidenceEstimator
from ..budget.controller import BudgetStopReason, RequestBudget
from ..budget.request_analyzer import RequestAnalyzer
from ..budget.scheduler import BudgetScheduler
from ..budget.safeguards import Safeguards
from ..budget.stopping import FinalizationResult, Finalizer
from ..budget.telemetry import BudgetRecord, BudgetTelemetry
from ..generation.generator import GenerationRequest, GenerationResult, Generator
from ..utils.errors import RequestError
from .execution_controller import ExecutionController, ExecutionOutcome
from .planner import ExecutionPlan, Planner
from .result_synthesizer import ResultSynthesizer, SynthesisResult
from .verification_pipeline import VerificationOutcome, VerificationStage

__all__ = ["InferenceRequest", "PipelineResult", "RequestPipeline"]


@dataclass
class InferenceRequest:
    """The internal representation of a request."""

    request_id: str = ""
    model_id: str = ""
    prompt: str = ""
    prompt_ids: list[int] = field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop: tuple[str, ...] = ()
    seed: int | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    latency_target_ms: float | None = None
    compute_budget: int | None = None
    tools: list[Any] = field(default_factory=list)
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "prompt_tokens": len(self.prompt_ids),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": list(self.stop),
            "seed": self.seed,
            "latency_target_ms": self.latency_target_ms,
            "compute_budget": self.compute_budget,
        }


@dataclass
class PipelineResult:
    """The result of a pipeline run."""

    request_id: str = ""
    text: str = ""
    finish_reason: str = "stop"
    generation: GenerationResult | None = None
    budget: RequestBudget | None = None
    finalization: FinalizationResult | None = None
    synthesis: SynthesisResult | None = None
    verification: VerificationOutcome | None = None
    plan: ExecutionPlan | None = None
    outcome: ExecutionOutcome | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    first_token_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "first_token_ms": None if self.first_token_ms is None else round(self.first_token_ms, 3),
            "budget": self.budget.to_dict() if self.budget else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "finalization": self.finalization.to_dict() if self.finalization else None,
        }


class RequestPipeline:
    """Runs a request through the adaptive budget layer."""

    def __init__(
        self,
        generator: Generator,
        allocator: BudgetAllocator | None = None,
        analyzer: RequestAnalyzer | None = None,
        complexity: ComplexityEstimator | None = None,
        scheduler: BudgetScheduler | None = None,
        planner: Planner | None = None,
        verification_stage: VerificationStage | None = None,
        synthesizer: ResultSynthesizer | None = None,
        safeguards: Safeguards | None = None,
        telemetry: BudgetTelemetry | None = None,
        resource_provider: Callable[[], ResourceState] | None = None,
        controller_description: Callable[[], dict[str, Any]] | None = None,
        prefix_finder: Callable[[InferenceRequest], Any] | None = None,
    ) -> None:
        self.generator = generator
        self.allocator = allocator or BudgetAllocator()
        self.analyzer = analyzer or RequestAnalyzer()
        self.complexity = complexity or ComplexityEstimator()
        self.scheduler = scheduler or BudgetScheduler()
        self.planner = planner or Planner()
        self.verification_stage = verification_stage or VerificationStage()
        self.synthesizer = synthesizer or ResultSynthesizer()
        self.safeguards = safeguards or Safeguards()
        self.telemetry = telemetry or BudgetTelemetry()
        self.resource_provider = resource_provider or (lambda: ResourceState())
        self.controller_description = controller_description or (lambda: {})
        self.prefix_finder = prefix_finder

    # ------------------------------------------------------------------
    def handle(
        self,
        request: InferenceRequest,
        cancelled: Callable[[], bool] | None = None,
        on_token: Callable[[Any], None] | None = None,
    ) -> PipelineResult:
        """Run a request through the whole pipeline."""

        started = time.perf_counter()
        result = PipelineResult(request_id=request.request_id)

        # --- stage 1: intake ------------------------------------------------
        analyzed = self.analyzer.analyze(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_tokens=len(request.prompt_ids),
            requested_output_tokens=self.safeguards.clamp_max_tokens(request.max_tokens),
            latency_target_ms=self.safeguards.clamp_latency_ms(request.latency_target_ms),
            requested_compute_budget=self.safeguards.clamp_compute_budget(request.compute_budget),
            requested_tools=request.tools,
        )

        resources = self.resource_provider()

        # --- stage 2: complexity --------------------------------------------
        estimate = self.complexity.estimate(analyzed, uncertainty=0.0, resource_pressure=resources.memory_pressure)

        # --- stage 3: allocation --------------------------------------------
        budget = self.allocator.allocate(
            analyzed, estimate, uncertainty=0.0, resources=resources, now=time.time()
        )

        # --- stage 4: planning ----------------------------------------------
        if budget.allocation.allow_planning and self.planner.should_plan(analyzed, estimate.score):
            result.plan = self.planner.create_plan(
                analyzed, estimate.score, max_tokens=budget.allocation.generation_tokens
            )

        # --- stage 5: execution ---------------------------------------------
        generation_request = GenerationRequest(
            prompt_ids=list(request.prompt_ids),
            max_tokens=budget.allocation.generation_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=tuple(request.stop),
            seed=request.seed,
            repetition_penalty=request.repetition_penalty,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            eos_token_id=self._eos_token_id(),
            deadline=budget.deadline,
        )
        controller = ExecutionController(self.generator, self.scheduler, ConfidenceEstimator())
        prefix = self._find_cached_prefix(request)
        outcome = controller.run(
            generation_request,
            budget,
            cancelled=cancelled,
            on_token=on_token,
            prefix=prefix,
        )
        result.outcome = outcome
        result.generation = outcome.result

        # --- stage 8: verification ------------------------------------------
        verification_passed: bool | None = None
        if budget.remaining_verification_passes > 0 or budget.consumed.verification_passes > 0:
            verification = self.verification_stage.run(
                outcome.text,
                task_category=analyzed.task_category,
                expected_json=bool(
                    (request.response_format or {}).get("type") == "json_object"
                ),
                passes_remaining=max(
                    budget.remaining_verification_passes, budget.consumed.verification_passes
                ),
            )
            result.verification = verification
            verification_passed = verification.result.passed

        # --- stage 9: finalisation ------------------------------------------
        synthesis = self.synthesizer.synthesize([outcome.text])
        finalization = self.finalizer.finalize(
            synthesis.text,
            budget,
            finish_reason=outcome.result.finish_reason if outcome.result else "stop",
            verification_passed=verification_passed,
            prompt_tokens=len(request.prompt_ids),
        )
        result.synthesis = synthesis
        result.finalization = finalization
        result.budget = budget
        result.text = synthesis.text
        result.finish_reason = finalization.finish_reason
        result.prompt_tokens = len(request.prompt_ids)
        result.completion_tokens = outcome.result.completion_tokens if outcome.result else 0
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        result.first_token_ms = outcome.result.first_token_ms if outcome.result else None

        # --- stage 10: telemetry --------------------------------------------
        self._record(budget, analyzed, estimate, result, resources)
        return result

    # ------------------------------------------------------------------
    def _find_cached_prefix(self, request: InferenceRequest) -> Any:
        """Return the cached prefix of a request, if there is one."""

        finder = getattr(self, "prefix_finder", None)
        if finder is None:
            return None
        try:
            return finder(request)
        except Exception:  # pragma: no cover - the cache must never break a request
            return None

    def _eos_token_id(self) -> int | None:
        tokenizer = getattr(self.generator, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            return int(tokenizer.eos_token_id)
        except Exception:  # pragma: no cover - defensive
            return None

    def _record(
        self,
        budget: RequestBudget,
        analyzed: Any,
        estimate: Any,
        result: PipelineResult,
        resources: ResourceState,
    ) -> None:
        features = self.allocator.build_features(
            analyzed,
            estimate,
            budget.uncertainty,
            resources,
            plan_steps=result.plan.length if result.plan else 0,
        )
        record = BudgetRecord(
            request_id=budget.request_id,
            task_category=analyzed.task_category,
            features=features,
            policy=budget.selected_policy,
            allocated_tokens=budget.allocation.generation_tokens,
            consumed_tokens=budget.consumed.generation_tokens,
            internal_steps=budget.consumed.internal_steps + 1,
            verification_runs=budget.consumed.verification_passes,
            verification_passed=(
                result.verification.result.passed if result.verification else None
            ),
            uncertainty_score=budget.uncertainty,
            complexity_score=budget.complexity,
            latency_ms=result.elapsed_ms,
        )
        self.telemetry.record(record)

    @property
    def finalizer(self) -> Finalizer:
        return Finalizer()
