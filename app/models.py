"""Schemas for structured output.

The split below is deliberate and worth keeping whatever the problem is.

Anything in ExtractedData is filled by the model, so every field is optional:
a model that omits a field should give you a null to handle, not a validation
crash. Anything in SystemAttachedData is filled by code and must never be
asked of the model — identity and timestamps that the model can write are
identity and timestamps an attacker can dictate through the prompt.
"""
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ExtractedData(BaseModel):
    """Placeholder for whatever the model is asked to produce."""

    answer: str | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SystemAttachedData(BaseModel):
    """Attached by code after the model has returned. Never prompted for."""

    user_id: str = Field(default="")
    request_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=_now)
