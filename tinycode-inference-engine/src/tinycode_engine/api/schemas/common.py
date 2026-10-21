"""Shared request schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Role", "ChatMessage", "StopSequence", "FunctionCall", "ToolDefinition"]


class Role(str):
    """The role of a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A single chat message."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(..., description="system, user, assistant or tool")
    content: str | None = Field(default=None, description="The message content.")
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_prompt_part(self) -> tuple[str, str]:
        """Return the ``(role, content)`` pair of the message."""

        return self.role, self.content or ""


StopSequence = str


class FunctionCall(BaseModel):
    """A function call in the OpenAI format."""

    name: str
    arguments: str = "{}"


class ToolDefinition(BaseModel):
    """A tool definition in the OpenAI format."""

    model_config = ConfigDict(extra="allow")

    type: str = "function"
    function: dict[str, Any] = Field(default_factory=dict)
