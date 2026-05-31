"""
FastAPI application entry point.
Exposes the trained pipeline as a REST API for production serving.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    MetricsResponse,
    ModelPrediction,
    PredictionRequest,
    PredictionResponse,
)
from src.config import cfg
from src.logging_setup import get_logger, setup_logging

log = get_logger("api")

# ─── App state ───────────────────────────────────────────────────────────────

_state: dict = {
    "feat": None,
    "regime": None,
    "sent": None,
    "trained": None,
    "metrics": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup."""
    setup_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_dir=cfg.paths.log_dir,
    )
    log.info("API starting up — loading pipeline artifacts")
    _load_artifacts()
    log.info("API ready")
    yield
    log.info("API shutting down")


def _load_artifacts() -> None:
    from src.models.fusion import TrainedFusion

    integration_csv = cfg.paths.output_dir / "integration_master.csv"
    metrics_json = cfg.paths.output_dir / "metrics_summary.json"

    if integration_csv.exists():
        df = pd.read_csv(integration_csv, index_col=0, parse_dates=True)
        _state["feat"] = df
        log.info("Integration CSV loaded", extra={"rows": len(df)})

    if metrics_json.exists():
        with open(metrics_json) as fh:
            _state["metrics"] = json.load(fh)

    model_dir = cfg.paths.model_dir
    if (model_dir / "fusion_meta.json").exists():
        try:
            _state["trained"] = TrainedFusion.load(model_dir)
            log.info("Fusion models loaded")
        except Exception as exc:
            log.error("Failed to load fusion models", extra={"error": str(exc)})


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Crisis Prediction System",
    description="Multimodal financial crisis early-warning API (HMM + GARCH + FinBERT + SHAP)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    feat = _state.get("feat")
    last_date = None
    if feat is not None and hasattr(feat.index, "max"):
        last_date = feat.index.max().date()
    return HealthResponse(
        status="ok",
        models_loaded=_state.get("trained") is not None,
        last_data_date=last_date,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Diagnostics"])
async def metrics() -> MetricsResponse:
    m = _state.get("metrics")
    if m is None:
        raise HTTPException(status_code=503, detail="Metrics not yet computed. Run the pipeline first.")
    return MetricsResponse(
        best_garch_label=m.get("best_garch", "unknown"),
        best_garch_bic=m.get("best_garch_bic", float("nan")),
        hmm_n_states=m.get("hmm_n_retained", -1),
        hmm_bic_profile={int(k): v for k, v in m.get("hmm_bic_profile", {}).items()},
        fsi_validity=m.get("fsi_validity", {}),
        fusion_best_f1_by_crisis=m.get("fusion_best_f1_by_crisis", {}),
        fusion_f1_target=m.get("fusion_f1_target", cfg.fusion.f1_target),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    return _make_prediction(request.target_date, request.horizon_days)


@app.post("/predict/batch", response_model=list, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> list:
    feat = _state.get("feat")
    if feat is None:
        raise HTTPException(status_code=503, detail="Pipeline data not loaded")

    dates = pd.date_range(request.start_date, request.end_date, freq="B")
    available = feat.index.intersection(pd.DatetimeIndex(dates))
    return [_make_prediction(d.date(), request.horizon_days) for d in available]


# ─── Private ─────────────────────────────────────────────────────────────────

def _make_prediction(target_date: date, horizon_days: int) -> PredictionResponse:
    feat = _state.get("feat")
    trained = _state.get("trained")
    warnings = []

    if feat is None:
        raise HTTPException(status_code=503, detail="Pipeline data not loaded. Run the training pipeline first.")

    ts = pd.Timestamp(target_date)
    if ts not in feat.index:
        # Find nearest available trading day
        avail = feat.index[feat.index <= ts]
        if avail.empty:
            raise HTTPException(status_code=404, detail=f"No data available on or before {target_date}")
        ts = avail[-1]
        warnings.append(f"Using nearest available trading day: {ts.date()}")

    row = feat.loc[ts]

    # Regime
    regime_raw = row.get("regime")
    regime_name = {0: "Stable", 1: "Volatile", 2: "Crisis"}.get(int(regime_raw), "Unknown") if regime_raw is not None else None
    regime_probs = {}
    for col, label in [("prob_stable", "Stable"), ("prob_volatile", "Volatile"), ("prob_crisis", "Crisis")]:
        if col in feat.columns:
            regime_probs[label] = round(float(row.get(col, 0.0)), 4)

    # Model predictions
    model_predictions = []
    ensemble_prob = None

    if trained is not None:
        feature_cols = trained.feature_cols
        available_cols = [c for c in feature_cols if c in feat.columns]
        missing_cols = [c for c in feature_cols if c not in feat.columns]
        if missing_cols:
            warnings.append(f"Missing feature columns: {missing_cols}")

        if available_cols:
            X = feat.loc[[ts], available_cols].fillna(0).values
            # Pad with zeros for any missing cols to maintain model input shape
            if len(available_cols) < len(feature_cols):
                import numpy as np
                padded = np.zeros((1, len(feature_cols)))
                col_idx = [feature_cols.index(c) for c in available_cols]
                padded[:, col_idx] = X
                X = padded

            probas = trained.predict_proba(X)
            probs_list = []
            for name, proba in probas.items():
                prob_val = float(proba[0])
                threshold = trained.thresholds.get(name, 0.5)
                model_predictions.append(ModelPrediction(
                    model_name=name,
                    crisis_probability=round(prob_val, 4),
                    crisis_predicted=prob_val >= threshold,
                    threshold=threshold,
                ))
                probs_list.append(prob_val)
            if probs_list:
                ensemble_prob = round(sum(probs_list) / len(probs_list), 4)

    return PredictionResponse(
        target_date=ts.date(),
        horizon_days=horizon_days,
        regime=regime_name,
        regime_probabilities=regime_probs if regime_probs else None,
        fsi=round(float(row.get("FSI", float("nan"))), 4) if "FSI" in feat.columns else None,
        fear_index=round(float(row.get("fear_index", float("nan"))), 4) if "fear_index" in feat.columns else None,
        model_predictions=model_predictions,
        ensemble_crisis_probability=ensemble_prob,
        data_as_of=feat.index.max().date(),
        warnings=warnings,
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=cfg.api.host,
        port=cfg.api.port,
        workers=cfg.api.workers,
        log_level=cfg.api.log_level,
        reload=False,
    )
