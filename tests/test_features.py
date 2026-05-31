"""
Unit tests for feature engineering.
All tests use synthetic data — no real market data required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestEngineerFeatures:
    def test_output_columns_present(self, feat_df):
        required = {"close", "log_ret", "vol_5d", "vol_21d", "vol_63d", "vol_126d",
                    "drawdown_63", "vix", "vix_chg", "vix_ma21", "vix_spike",
                    "mom_5d", "mom_21d", "mom_63d", "vol_ratio"}
        assert required <= set(feat_df.columns)

    def test_log_returns_are_finite(self, feat_df):
        ret = feat_df["log_ret"].dropna()
        assert np.isfinite(ret).all(), "log_ret contains non-finite values"

    def test_vol_is_non_negative(self, feat_df):
        for col in ["vol_5d", "vol_21d", "vol_63d"]:
            assert (feat_df[col].dropna() >= 0).all(), f"{col} has negative values"

    def test_drawdown_is_non_positive(self, feat_df):
        dd = feat_df["drawdown_63"].dropna()
        assert (dd <= 0).all(), "drawdown_63 should be <= 0"

    def test_vix_spike_is_binary(self, feat_df):
        vals = feat_df["vix_spike"].dropna().unique()
        assert set(vals).issubset({0, 1}), "vix_spike must be binary"

    def test_index_is_monotonic(self, feat_df):
        assert feat_df.index.is_monotonic_increasing

    def test_no_duplicate_dates(self, feat_df):
        assert not feat_df.index.duplicated().any()

    def test_missing_rate_acceptable(self, feat_df):
        missing = feat_df["log_ret"].isna().mean()
        assert missing < 0.05, f"log_ret has {missing:.1%} missing — too high"


class TestValidateFeatures:
    def test_valid_frame_passes(self, feat_df):
        from src.features.engineering import validate_features
        validate_features(feat_df)  # should not raise

    def test_missing_column_raises(self, feat_df):
        from src.features.engineering import validate_features
        bad = feat_df.drop(columns=["log_ret"])
        with pytest.raises(ValueError, match="log_ret"):
            validate_features(bad)

    def test_non_monotonic_index_raises(self, feat_df):
        from src.features.engineering import validate_features
        shuffled = feat_df.sample(frac=1, random_state=0)
        with pytest.raises(ValueError, match="not sorted"):
            validate_features(shuffled)
