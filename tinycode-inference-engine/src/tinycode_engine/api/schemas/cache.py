"""Request schemas of the cache endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import ChatMessage

__all__ = ["CacheInputRequest"]


class CacheInputRequest(BaseModel):
    """The body of ``POST /v1/cache/input``."""

    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, description="The model the input belongs to.")
    input: str | None = Field(default=None, description="The text to cache.")
    messages: list[ChatMessage] | None = Field(
        default=None, description="A chat conversation to cache (rendered like a chat prompt)."
    )
    prefix: bool = Field(
        default=True,
        description="Also compute and store the KV state of the input.",
    )

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("'input' must not be empty.")
        return value
