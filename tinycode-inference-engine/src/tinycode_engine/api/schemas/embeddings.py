"""Request schemas of the embeddings and responses endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["EmbeddingsRequest", "ResponsesRequest"]


class EmbeddingsRequest(BaseModel):
    """The body of ``POST /v1/embeddings``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model to use.")
    input: str | list[str] = Field(..., description="The input text(s).")
    encoding_format: str = Field(default="float")
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None

    @field_validator("encoding_format")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        if value not in {"float", "base64"}:
            raise ValueError("'encoding_format' must be 'float' or 'base64'.")
        return value


class ResponsesRequest(BaseModel):
    """The body of ``POST /v1/responses``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model to use.")
    input: str | list[dict[str, Any]] | list[str] = Field(..., description="The input.")
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1, le=1024)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool | None = None
    store: bool | None = None
    metadata: dict[str, Any] | None = None
