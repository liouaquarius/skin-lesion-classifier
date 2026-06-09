"""Pydantic schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class ExplainResponse(PredictionResponse):
    grad_cam_image: str | None = None
