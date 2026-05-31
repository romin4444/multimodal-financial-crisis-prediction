"""
Pydantic request/response schemas for the prediction API.
All user-facing inputs are validated here — the models never see raw dicts.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Request schemas ─────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Request a crisis probability estimate for a specific date."""
    target_date: date = Field(..., description="Date to predict crisis probability for (YYYY-MM-DD)")
    horizon_days: int = Field(5, ge=1, le=60, description="Prediction horizon in trading days")

    @field_validator("target_date")
    @classmethod
    def not_in_future(cls, v: date) -> date:
        from datetime import date as date_
        if v > date_.today():
            raise ValueError("target_date cannot be in the future")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction for a date range."""
    start_date: date = Field(..., description="Start of prediction window")
    end_date: date = Field(..., description="End of prediction window")
    horizon_days: int = Field(5, ge=1, le=60)

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


# ─── Response schemas ────────────────────────────────────────────────────────

class ModelPrediction(BaseModel):
    model_name: str
    crisis_probability: float = Field(..., ge=0.0, le=1.0)
    crisis_predicted: bool
    threshold: float


class PredictionResponse(BaseModel):
    target_date: date
    horizon_days: int
    regime: Optional[str] = None  # "Stable", "Volatile", or "Crisis"
    regime_probabilities: Optional[Dict[str, float]] = None
    fsi: Optional[float] = Field(None, ge=0.0, le=1.0, description="Financial Stress Index [0,1]")
    fear_index: Optional[float] = Field(None, ge=0.0, le=1.0, description="FinBERT fear index [0,1]")
    model_predictions: List[ModelPrediction] = []
    ensemble_crisis_probability: Optional[float] = None
    data_as_of: Optional[date] = None
    warnings: List[str] = []


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    last_data_date: Optional[date] = None
    pipeline_version: str = "2.0.0"


class MetricsResponse(BaseModel):
    best_garch_label: str
    best_garch_bic: float
    hmm_n_states: int
    hmm_bic_profile: Dict[int, float]
    fsi_validity: dict
    fusion_best_f1_by_crisis: Dict[str, float]
    fusion_f1_target: float
