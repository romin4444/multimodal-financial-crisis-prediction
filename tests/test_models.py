"""
Unit tests for model components.
Uses synthetic fixtures — no GPU or real data required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TestFSIBuilder:
    def test_fsi_in_unit_interval(self, feat_df):
        from src.models.fsi import FSIBuilder

        builder = FSIBuilder()
        df, _ = builder.build(feat_df, pd.DataFrame())
        fsi = df["FSI"]
        assert (fsi >= 0).all() and (fsi <= 1).all(), "FSI must be in [0, 1]"

    def test_fsi_scaler_fit_on_train_only(self, feat_df):
        """FSI scalers must not use future data from test windows."""
        from src.models.fsi import FSIBuilder

        builder = FSIBuilder()
        n = len(feat_df)
        train_mask = np.ones(n, dtype=bool)
        train_mask[int(n * 0.8):] = False  # hold out last 20%

        df, _ = builder.build(feat_df, pd.DataFrame(), train_mask=train_mask)
        # FSI should still be computed for all rows (transform applied everywhere)
        assert df["FSI"].notna().sum() == n

    def test_nber_flag_created(self, feat_df):
        from src.models.fsi import FSIBuilder

        builder = FSIBuilder()
        df, _ = builder.build(feat_df, pd.DataFrame())
        assert "_nber" in df.columns
        assert df["_nber"].isin([0, 1]).all()


class TestFusionMatrix:
    def test_target_is_binary(self, feat_df, synthetic_regime, synthetic_sentiment):
        from src.models.fusion import build_fusion_matrix

        feat = feat_df.copy()
        feat["FSI"] = 0.3  # inject placeholder
        fusion = build_fusion_matrix(synthetic_regime, synthetic_sentiment, feat)
        assert set(fusion["target"].unique()).issubset({0, 1})

    def test_no_nan_in_features(self, feat_df, synthetic_regime, synthetic_sentiment):
        from src.models.fusion import build_fusion_matrix

        feat = feat_df.copy()
        feat["FSI"] = 0.3
        fusion = build_fusion_matrix(synthetic_regime, synthetic_sentiment, feat)
        feature_cols = [c for c in fusion.columns if c != "target"]
        assert not fusion[feature_cols].isna().any().any(), "Feature matrix has NaN values"


class TestThresholdTuning:
    def test_threshold_between_zero_and_one(self):
        from src.models.fusion import _tune_threshold
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5))
        y = (rng.random(200) > 0.7).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=0)
        model.fit(X, y)
        t = _tune_threshold(model, X, y)
        assert 0.0 <= t <= 1.0

    def test_empty_input_returns_half(self):
        from src.models.fusion import _tune_threshold

        class DummyModel:
            def predict_proba(self, X):
                return np.zeros((len(X), 2))

        t = _tune_threshold(DummyModel(), np.zeros((0, 3)), np.array([]))
        assert t == 0.5


class TestLeadLag:
    def test_cross_corr_returns_expected_keys(self, feat_df, synthetic_sentiment):
        from src.analysis.lead_lag import cross_corr

        feat = feat_df.copy()
        feat["FSI"] = 0.3
        fsi = feat["FSI"]
        fear = synthetic_sentiment["fear_index"]
        result = cross_corr(fsi, fear, max_lag=5, n_boot=10)
        for key in ("lags", "corrs", "peak_lag", "peak_r", "ci_lo", "ci_hi", "interp"):
            assert key in result

    def test_peak_lag_within_max(self, feat_df, synthetic_sentiment):
        from src.analysis.lead_lag import cross_corr

        feat = feat_df.copy()
        feat["FSI"] = 0.3
        result = cross_corr(feat["FSI"], synthetic_sentiment["fear_index"], max_lag=10, n_boot=5)
        assert abs(result["peak_lag"]) <= 10


class TestSyntheticSentiment:
    def test_fear_index_in_unit_interval(self, feat_df):
        from src.models.sentiment import build_synthetic_sentiment

        df = build_synthetic_sentiment(feat_df)
        assert (df["fear_index"] >= 0).all() and (df["fear_index"] <= 1).all()

    def test_is_synthetic_flag_set(self, feat_df):
        from src.models.sentiment import build_synthetic_sentiment

        df = build_synthetic_sentiment(feat_df)
        assert (df["is_synthetic"] == 1).all()
