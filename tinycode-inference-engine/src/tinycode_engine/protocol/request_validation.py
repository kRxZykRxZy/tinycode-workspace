"""Validation of the OpenAI compatible request fields."""

from __future__ import annotations

from typing import Any

from ..utils.errors import RequestError

__all__ = [
    "validate_temperature",
    "validate_top_p",
    "validate_max_tokens",
    "validate_stop",
    "validate_stream",
    "validate_seed",
    "validate_penalties",
    "validate_model_name",
    "VALID_FINISH_REASONS",
]

VALID_FINISH_REASONS = ("stop", "length", "content_filter", "tool_calls", "function_call")


def validate_model_name(model: Any) -> str:
    """Validate the requested model name."""

    if model is None:
        raise RequestError("The 'model' field is required.")
    if not isinstance(model, str) or not model.strip():
        raise RequestError("The 'model' field must be a non-empty string.")
    return model


def validate_temperature(value: Any) -> float:
    """Validate the temperature."""

    if value is None:
        return 1.0
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise RequestError("'temperature' must be a number.") from exc
    if temperature < 0.0 or temperature > 2.0:
        raise RequestError("'temperature' must be between 0 and 2.")
    return temperature


def validate_top_p(value: Any) -> float:
    """Validate the top_p value."""

    if value is None:
        return 1.0
    try:
        top_p = float(value)
    except (TypeError, ValueError) as exc:
        raise RequestError("'top_p' must be a number.") from exc
    if top_p <= 0.0 or top_p > 1.0:
        raise RequestError("'top_p' must be greater than 0 and at most 1.")
    return top_p


def validate_max_tokens(value: Any, default: int = 256) -> int:
    """Validate the requested number of output tokens."""

    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError("'max_tokens' must be an integer.")
    if value <= 0:
        raise RequestError("'max_tokens' must be greater than zero.")
    if value > 1024:
        raise RequestError(
            "'max_tokens' must not exceed 1024; the context length is fixed at 1024 tokens."
        )
    return int(value)


def validate_stop(value: Any) -> tuple[str, ...]:
    """Validate the stop sequences."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, str):
                raise RequestError("'stop' must contain strings.")
        return tuple(value)
    raise RequestError("'stop' must be a string or a list of strings.")


def validate_stream(value: Any) -> bool:
    """Validate the stream flag."""

    if value is None:
        return False
    if not isinstance(value, bool):
        raise RequestError("'stream' must be a boolean.")
    return value


def validate_seed(value: Any) -> int | None:
    """Validate the seed."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError("'seed' must be an integer.")
    return int(value)


def validate_penalties(
    presence_penalty: Any = None, frequency_penalty: Any = None, repetition_penalty: Any = None
) -> tuple[float, float, float]:
    """Validate the penalty fields."""

    def _check(value: Any, name: str, low: float, high: float) -> float:
        if value is None:
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RequestError(f"'{name}' must be a number.") from exc
        if number < low or number > high:
            raise RequestError(f"'{name}' must be between {low} and {high}.")
        return number

    presence = _check(presence_penalty, "presence_penalty", -2.0, 2.0)
    frequency = _check(frequency_penalty, "frequency_penalty", -2.0, 2.0)
    repetition = _check(repetition_penalty, "repetition_penalty", 0.0, 2.0) or 1.0
    return presence, frequency, repetition
