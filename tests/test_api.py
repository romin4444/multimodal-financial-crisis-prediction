"""
API integration tests.
Uses FastAPI's TestClient — no real server needed.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.app import app, _state

    # Inject minimal mock state so endpoints work without a full pipeline run
    idx = pd.bdate_range("2010-01-01", "2020-12-31", freq="B")
    df = pd.DataFrame(
        {
            "regime": 0,
            "prob_stable": 0.7,
            "prob_volatile": 0.2,
            "prob_crisis": 0.1,
            "FSI": 0.3,
            "fear_index": 0.25,
        },
        index=idx,
    )
    _state["feat"] = df
    _state["metrics"] = {
        "best_garch": "GARCH(1,1)",
        "best_garch_bic": -12345.0,
        "hmm_n_retained": 3,
        "hmm_bic_profile": {2: -9000.0, 3: -9500.0, 4: -9800.0},
        "fsi_validity": {"stlfsi_r": 0.72, "nber_roc_auc": 0.85},
        "fusion_best_f1_by_crisis": {"GFC_2008": 0.75, "COVID_2020": 0.80},
        "fusion_f1_target": 0.70,
        "finbert_vs_vader": {},
    }

    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


class TestMetrics:
    def test_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/metrics").json()
        assert "best_garch_label" in data
        assert "fusion_f1_target" in data
        assert "fusion_best_f1_by_crisis" in data


class TestPredict:
    def test_valid_date_returns_200(self, client):
        response = client.post("/predict", json={"target_date": "2015-06-01", "horizon_days": 5})
        assert response.status_code == 200

    def test_response_has_regime(self, client):
        data = client.post("/predict", json={"target_date": "2015-06-01"}).json()
        assert "regime" in data

    def test_future_date_rejected(self, client):
        future = (date.today() + timedelta(days=30)).isoformat()
        response = client.post("/predict", json={"target_date": future})
        assert response.status_code == 422

    def test_invalid_horizon_rejected(self, client):
        response = client.post("/predict", json={"target_date": "2015-06-01", "horizon_days": 999})
        assert response.status_code == 422

    def test_weekend_date_returns_nearest_trading_day(self, client):
        # 2015-06-06 is a Saturday
        data = client.post("/predict", json={"target_date": "2015-06-06"}).json()
        assert data["target_date"] != "2015-06-06"  # adjusted to Friday or Monday
        assert "warnings" in data
        assert len(data["warnings"]) > 0


class TestBatchPredict:
    def test_returns_list(self, client):
        data = client.post(
            "/predict/batch",
            json={"start_date": "2015-01-02", "end_date": "2015-01-09", "horizon_days": 5},
        ).json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_end_before_start_rejected(self, client):
        response = client.post(
            "/predict/batch",
            json={"start_date": "2015-06-01", "end_date": "2015-05-01"},
        )
        assert response.status_code == 422
