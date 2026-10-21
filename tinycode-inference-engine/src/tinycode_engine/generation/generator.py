"""The generation loop.

The generator owns the autoregressive decoding: it feeds the prompt into
the model, then decodes token by token while honouring the stop
conditions, the cancellation flag and the computation budget.

``GenerationSession`` exposes the loop step by step so that the scheduler
can interleave several requests (request level continuous batching).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from ..cache.kv_cache import KVCache
from ..utils.errors import ModelError
from .greedy import select_greedy
from .logits_processing import LogitsProcessors
from .sampling import sample_token
from .stopping import StopCondition, StopController, StopReason

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "GenerationSession",
    "Generator",
    "TokenChunk",
]


@dataclass
class GenerationRequest:
    """Everything the generator needs for one completion."""

    prompt_ids: list[int]
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop_sequences: tuple[str, ...] = ()
    seed: int | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    eos_token_id: int | None = None
    deadline: float | None = None
    max_internal_steps: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": list(self.stop_sequences),
            "seed": self.seed,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "eos_token_id": self.eos_token_id,
            "max_internal_steps": self.max_internal_steps,
        }


@dataclass
class GenerationResult:
    """The outcome of a generation."""

    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = StopReason.LENGTH
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    first_token_ms: float | None = None
    stop_sequence: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "first_token_ms": None if self.first_token_ms is None else round(self.first_token_ms, 3),
            "stop_sequence": self.stop_sequence,
        }


@dataclass
class TokenChunk:
    """A streamed chunk."""

    text: str
    token_id: int
    index: int
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_id": self.token_id,
            "index": self.index,
            "finish_reason": self.finish_reason,
        }


class GenerationSession:
    """A step-wise generation session.

    The session can be advanced one token at a time, which allows the
    scheduler to interleave several requests.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        request: GenerationRequest | None = None,
        cache: KVCache | None = None,
        prefix: Any = None,
        backend: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.request = request
        self.cache = cache or self._new_cache()
        self.prefix = prefix
        self.prefix_tokens = 0
        self.backend = backend
        # A backend that cannot keep a KV cache has to recompute the whole
        # sequence for every token; the session keeps the token list.
        self.use_cache = backend is None or bool(backend.capabilities().supports_kv_cache)
        self.rng = np.random.default_rng(request.seed) if request and request.seed is not None else None
        self.processors = LogitsProcessors(
            repetition_penalty=request.repetition_penalty if request else 1.0,
            frequency_penalty=request.frequency_penalty if request else 0.0,
            presence_penalty=request.presence_penalty if request else 0.0,
        )
        self.controller = StopController(
            StopCondition(
                max_tokens=request.max_tokens if request else 128,
                stop_sequences=tuple(request.stop_sequences) if request else (),
                eos_token_id=request.eos_token_id if request else None,
                deadline=request.deadline if request else None,
            )
        )
        self.generated: list[int] = []
        self.text_parts: list[str] = []
        self.logits_history: list[Any] = []
        self.finished = False
        self.finish_reason: str | None = None
        self.stop_sequence: str | None = None
        self.started = time.perf_counter()
        self.first_token_ms: float | None = None
        self._pending_logits: Any = None
        self._prefilled = False

    # ------------------------------------------------------------------
    def _new_cache(self) -> KVCache:
        capabilities = self.model.capabilities()
        config = self.model.config
        return KVCache(
            n_layers=int(config["n_layers"]),
            n_heads=int(config["n_heads"]),
            head_dim=int(config["d_model"]) // int(config["n_heads"]),
            context_length=int(capabilities.context_length),
        )

    def _decode(self, token_ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    # ------------------------------------------------------------------
    def prefill(self) -> None:
        """Run the model over the prompt.

        When a cached prefix is available and it matches the beginning of
        the prompt, the cached KV state is restored and only the remaining
        tokens are computed.
        """

        if self._prefilled:
            return
        if self.request is None or not self.request.prompt_ids:
            raise ModelError("The prompt is empty.")
        prompt_ids = list(self.request.prompt_ids)
        if not self.cache.can_append(len(prompt_ids)):
            raise ModelError(
                f"The prompt of {len(prompt_ids)} tokens does not fit into the fixed "
                f"{self.cache.context_length} token context."
            )

        restored = self._restore_prefix(prompt_ids)
        remaining = prompt_ids[restored:]
        if not remaining:
            # The cached prefix covers the whole prompt: the logits of the
            # last position were cached together with the KV state.
            payload = getattr(self.prefix, "payload", None) or {}
            logits = payload.get("logits")
            if logits is None:
                raise ModelError(
                    "The cached prefix covers the whole prompt but does not contain "
                    "the logits of the last position."
                )
            self._pending_logits = logits
            self._prefilled = True
            return
        self._pending_logits = self._forward(remaining)
        self._prefilled = True

    def _forward(self, token_ids: Any) -> Any:
        """Run the model (or the backend) over ``token_ids``."""

        if self.backend is not None:
            cache = self.cache if self.use_cache else None
            return self.backend.execute_forward(token_ids, cache=cache)
        return self.model.forward(token_ids, cache=self.cache)

    def _restore_prefix(self, prompt_ids: list[int]) -> int:
        """Restore a cached prefix into the KV cache.

        Returns:
            The number of tokens that were restored.
        """

        prefix = self.prefix
        if prefix is None:
            return 0
        payload = getattr(prefix, "payload", None)
        tokens = list(getattr(prefix, "token_ids", []) or [])
        if not payload or not tokens:
            return 0
        if list(prompt_ids[: len(tokens)]) != tokens:
            return 0
        length = int(payload.get("length", len(tokens)))
        keys = payload.get("keys") or []
        values = payload.get("values") or []
        if len(keys) < self.cache.n_layers or len(values) < self.cache.n_layers:
            return 0
        for layer in range(self.cache.n_layers):
            self.cache.keys[layer][:, :length, :] = keys[layer]
            self.cache.values[layer][:, :length, :] = values[layer]
        self.cache.advance(length)
        self.prefix_tokens = length
        return length

    def step(self) -> TokenChunk | None:
        """Generate the next token.

        Returns:
            The produced chunk, or ``None`` when the session is finished.
        """

        if self.finished:
            return None
        if not self._prefilled:
            self.prefill()

        request = self.request
        assert request is not None

        logits = self.processors.apply(self._pending_logits, self.generated)
        self.logits_history.append(logits)

        if request.temperature <= 0:
            token_id = select_greedy(logits)
        else:
            token_id = sample_token(
                logits,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                rng=self.rng,
            )

        self.generated.append(token_id)
        text = self._decode([token_id])
        self.text_parts.append(text)

        if self.first_token_ms is None:
            self.first_token_ms = (time.perf_counter() - self.started) * 1000.0

        reason = self.controller.update(token_id, "".join(self.text_parts), time.time())
        if reason:
            self.finished = True
            self.finish_reason = reason
            for sequence in request.stop_sequences:
                if sequence and sequence in "".join(self.text_parts):
                    self.stop_sequence = sequence
                    break
            return TokenChunk(
                text=text, token_id=token_id, index=len(self.generated) - 1, finish_reason=reason
            )

        if self.cache.full():
            self.finished = True
            self.finish_reason = StopReason.LENGTH
            return TokenChunk(
                text=text,
                token_id=token_id,
                index=len(self.generated) - 1,
                finish_reason=self.finish_reason,
            )

        if self.use_cache:
            self._pending_logits = self._forward(np.asarray([token_id], dtype=np.int64))
        else:
            context = list(self.request.prompt_ids) + list(self.generated)
            self._pending_logits = self._forward(np.asarray(context, dtype=np.int64))
        return TokenChunk(text=text, token_id=token_id, index=len(self.generated) - 1)

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def result(self) -> GenerationResult:
        """Return the result of the session."""

        text = self.text
        if self.stop_sequence and self.stop_sequence in text:
            text = text.split(self.stop_sequence)[0]
        prompt_tokens = len(self.request.prompt_ids) if self.request else 0
        return GenerationResult(
            text=text,
            token_ids=list(self.generated),
            finish_reason=self.finish_reason or StopReason.LENGTH,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(self.generated),
            elapsed_ms=(time.perf_counter() - self.started) * 1000.0,
            first_token_ms=self.first_token_ms,
            stop_sequence=self.stop_sequence,
        )

    def release(self) -> None:
        """Free the cache of the session."""

        self.cache.release()


class Generator:
    """Runs the autoregressive decoding loop."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        cache_factory: Callable[[], KVCache] | None = None,
        backend: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._cache_factory = cache_factory
        self.backend = backend

    # ------------------------------------------------------------------
    def new_session(self, request: GenerationRequest, prefix: Any = None) -> GenerationSession:
        """Create a step-wise session for a request."""

        cache = self._cache_factory() if self._cache_factory else None
        return GenerationSession(
            self.model, self.tokenizer, request, cache=cache, prefix=prefix, backend=self.backend
        )

    def generate(
        self,
        request: GenerationRequest,
        cancelled: Callable[[], bool] | None = None,
        on_token: Callable[[TokenChunk], None] | None = None,
        on_logits: Callable[[Any, list[int]], Any] | None = None,
        budget_check: Callable[[int], str | None] | None = None,
        prefix: Any = None,
    ) -> GenerationResult:
        """Generate a completion."""

        if not request.prompt_ids:
            raise ModelError("The prompt is empty.")

        session = self.new_session(request, prefix=prefix)
        try:
            session.prefill()
            while True:
                if cancelled is not None and cancelled():
                    session.finished = True
                    session.finish_reason = StopReason.CANCELLED
                    break
                chunk = session.step()
                if chunk is None:
                    break
                if on_token is not None:
                    on_token(chunk)
                if on_logits is not None and session.logits_history:
                    on_logits(session.logits_history[-1], session.generated)
                if budget_check is not None:
                    budget_reason = budget_check(len(session.generated))
                    if budget_reason:
                        session.finish_reason = budget_reason
                        session.finished = True
                        break
            return session.result()
        finally:
            session.release()

    def stream(
        self,
        request: GenerationRequest,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[TokenChunk]:
        """Yield the generated tokens as they are produced."""

        session = self.new_session(request)
        try:
            session.prefill()
            while True:
                if cancelled is not None and cancelled():
                    break
                chunk = session.step()
                if chunk is None:
                    break
                yield chunk
        finally:
            session.release()
