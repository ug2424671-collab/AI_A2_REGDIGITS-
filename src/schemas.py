"""Typed request/response models shared by predict.py and the FastAPI service."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FarmRecord(BaseModel):
    """One farm record. Every field is required; ranges match the training data rules."""
    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(ge=600, le=1400)
    avg_temp_c: float = Field(ge=12, le=26)
    soil_moisture_pct: float = Field(ge=20, le=80)
    fertilizer_kg: float = Field(ge=0, le=250)
    field_size_ha: float = Field(ge=0.2, le=4.0)
    season: Literal["A", "B"]
    cooperative_support: int = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    yield_tons_estimate: float
    high_risk_prediction: int
    high_risk_label: str
    high_risk_probability: float
    cluster: int
    cluster_profile: str
    verification_code: str
    disclaimer: str
