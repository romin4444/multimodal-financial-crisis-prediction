"""
TRUE end-to-end API integration test.

Unlike tests/test_api.py (which injects a mocked in-memory _state), this test
exercises the REAL wiring: it trains a real model, saves it through the real
joblib serializer (TrainedFusion.save), writes a real integration_master.csv,
then loads everything back through the REAL loader (_load_artifacts) and serves
a prediction. This catches artifact/loader/serializer connection bugs that the
mocked unit tests cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier


def test_real_artifact_roundtrip_and_predict(tmp_path, monkeypatch):
    from src.config import cfg
    from src.models.fusion import FEATURE_COLS, TrainedFusion
    import src.api.app as appmod

    # Redirect config paths to a temp workspace
    out_dir = tmp_path
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(cfg.paths, "output_dir", out_dir)
    monkeypatch.setattr(cfg.paths, "model_dir", model_dir)

    # ── Build a small but REAL fusion training frame over FEATURE_COLS ──
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=400)
    fcols = list(FEATURE_COLS)
    X = pd.DataFrame(rng.random((len(idx), len(fcols))), columns=fcols, index=idx)
    # A learnable signal so the model is non-degenerate
    y = ((X["FSI"] + X["fear_index"]) > 1.0).astype(int).to_numpy()
    y[:15] = 1  # guarantee positives

    rf = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=0).fit(X.values, y)
    trained = TrainedFusion(
        models={"Random Forest": rf},
        thresholds={"Random Forest": 0.5},
        feature_cols=fcols,
        train_size=len(idx),
        val_size=0,
    )
    # REAL serialization path (joblib + meta json)
    trained.save(model_dir)
    assert (model_dir / "fusion_meta.json").exists()
    assert any(model_dir.glob("fusion_*.joblib"))

    # ── Write a REAL integration_master.csv the API will load ──────────
    integ = X.copy()
    integ["regime"] = rng.integers(0, 3, len(idx))
    integ.to_csv(out_dir / "integration_master.csv")

    # ── Load via the REAL loader (not the mocked _state) ───────────────
    appmod._state = {"feat": None, "regime": None, "sent": None, "trained": None, "metrics": None}
    appmod._load_artifacts()
    assert appmod._state["trained"] is not None, "TrainedFusion.load did not wire up"
    assert appmod._state["feat"] is not None, "integration_master.csv did not load"

    # ── Serve a real prediction through the full route ─────────────────
    with TestClient(appmod.app) as client:
        resp = client.post("/predict", json={"target_date": "2015-06-01", "horizon_days": 5})
        assert resp.status_code == 200
        body = resp.json()
        # A genuine model prediction came back through the real serialization path
        assert len(body["model_predictions"]) >= 1
        pred = body["model_predictions"][0]
        assert pred["model_name"] == "Random Forest"
        assert 0.0 <= pred["crisis_probability"] <= 1.0
        # No "missing feature columns" warning — integration CSV had all features
        assert not any("Missing feature" in w for w in body.get("warnings", []))
        # Honesty: uncalibrated models must be flagged AND warned about, never
        # served as literal probabilities silently.
        assert body["probabilities_calibrated"] is False
        assert any("calibrated" in w.lower() for w in body.get("warnings", []))


def test_real_artifact_health_reports_loaded(tmp_path, monkeypatch):
    """/health should report models_loaded=True once real artifacts are present."""
    from src.config import cfg
    from src.models.fusion import FEATURE_COLS, TrainedFusion
    import src.api.app as appmod

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(cfg.paths, "output_dir", tmp_path)
    monkeypatch.setattr(cfg.paths, "model_dir", model_dir)

    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2016-01-01", periods=200)
    fcols = list(FEATURE_COLS)
    X = pd.DataFrame(rng.random((len(idx), len(fcols))), columns=fcols, index=idx)
    y = (rng.random(len(idx)) > 0.7).astype(int)
    y[:10] = 1
    rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X.values, y)
    TrainedFusion(models={"Random Forest": rf}, thresholds={"Random Forest": 0.5},
                  feature_cols=fcols, train_size=len(idx), val_size=0).save(model_dir)
    integ = X.copy()
    integ["regime"] = 0
    integ.to_csv(tmp_path / "integration_master.csv")

    appmod._state = {"feat": None, "regime": None, "sent": None, "trained": None, "metrics": None}
    with TestClient(appmod.app) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok"
        assert h["models_loaded"] is True
