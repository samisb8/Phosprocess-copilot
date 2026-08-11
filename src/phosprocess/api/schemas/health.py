"""Schemas for API health endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Response returned when the API process is running."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
