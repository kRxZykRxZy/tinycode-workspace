"""Request schemas of the completions endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import ChatMessage, ToolDefinition

__all__ = ["CompletionRequest", "ChatCompletionRequest", "ResponseFormat"]


class ResponseFormat(BaseModel):
    """The ``response_format`` field."""

    model_config = ConfigDict(extra="allow")

    type: str = "text"


class CompletionRequest(BaseModel):
    """The body of ``POST /v1/completions``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model to use.")
    prompt: str | list[str] = Field(..., description="The prompt (a string or a list of strings).")
    max_tokens: int | None = Field(default=None, ge=1, le=1024)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    stream: bool | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    seed: int | None = None
    logprobs: int | None = None
    echo: bool | None = None
    user: str | None = None

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("'prompt' must not be empty.")
            return value
        if not value:
            raise ValueError("'prompt' must not be empty.")
        for item in value:
            if not isinstance(item, str):
                raise ValueError("'prompt' must contain strings.")
        return value


class ChatCompletionRequest(BaseModel):
    """The body of ``POST /v1/chat/completions``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model to use.")
    messages: list[ChatMessage] = Field(..., min_length=1, description="The conversation.")
    max_tokens: int | None = Field(default=None, ge=1, le=1024)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=1024)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    stream: bool | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Any | None = None
    response_format: ResponseFormat | None = None
    user: str | None = None

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        for message in value:
            if not message.role:
                raise ValueError("every message needs a role.")
            if message.content is None and not message.tool_calls:
                raise ValueError("every message needs content or tool calls.")
        return value

    def effective_max_tokens(self) -> int:
        """Return the requested output budget, honouring both field names."""

        if self.max_completion_tokens is not None:
            return int(self.max_completion_tokens)
        if self.max_tokens is not None:
            return int(self.max_tokens)
        return 256
