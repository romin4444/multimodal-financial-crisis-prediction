"""
Unit tests for the stock-direction (up/down) detection module.
Asserts correct labeling, causal right-edge handling, baseline behavior,
and metric sanity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TestDirectionLabel:
    def test_up_market_labeled_up(self):
        from src.v3.direction import direction_label
        idx = pd.bdate_range("2010-01-01", periods=60)
        close = pd.Series(np.linspace(100, 160, 60), index=idx)
        label = direction_label(close, horizon=5)
        # strictly rising → every resolvable day is 'up'
        assert label.dropna().min() == 1.0

    def test_down_market_labeled_down(self):
        from src.v3.direction import direction_label
        idx = pd.bdate_range("2010-01-01", periods=60)
        close = pd.Series(np.linspace(160, 100, 60), index=idx)
        label = direction_label(close, horizon=5)
        assert label.dropna().max() == 0.0

    def test_right_edge_is_nan(self):
        from src.v3.direction import direction_label
        idx = pd.bdate_range("2010-01-01", periods=30)
        close = pd.Series(np.full(30, 100.0), index=idx)
        label = direction_label(close, horizon=5)
        # last `horizon` rows cannot resolve → NaN, never fabricated
        assert label.iloc[-5:].isna().all()

    def test_summary_keys(self):
        from src.v3.direction import direction_label, direction_summary
        idx = pd.bdate_range("2010-01-01", periods=100)
        rng = np.random.default_rng(0)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))), index=idx)
        s = direction_summary(direction_label(close, horizon=5))
        assert {"n", "n_up", "up_rate"} <= set(s)


class TestStockFeatures:
    def test_features_present_and_finite(self):
        from src.v3.direction import build_stock_features, STOCK_FEATURE_COLS
        idx = pd.bdate_range("2010-01-01", periods=400)
        rng = np.random.default_rng(1)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
        stk = pd.DataFrame({"Close": close, "Volume": rng.integers(1e6, 5e6, 400)}, index=idx)
        ctx = pd.DataFrame({"mkt_vix": rng.uniform(10, 40, 400)}, index=idx)
        feats = build_stock_features(stk, ctx)
        for c in STOCK_FEATURE_COLS:
            assert c in feats.columns
        # after warmup, no infinities
        tail = feats[STOCK_FEATURE_COLS].iloc[250:]
        assert np.isfinite(tail.to_numpy()).all()

    def test_rsi_bounded(self):
        from src.v3.direction import _rsi
        idx = pd.bdate_range("2010-01-01", periods=200)
        rng = np.random.default_rng(2)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=idx)
        rsi = _rsi(close, 14).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()


class TestBaselines:
    def test_always_up_predicts_one(self):
        from src.v3.direction import AlwaysUpPredictor
        X = pd.DataFrame({"a": range(50)})
        p = AlwaysUpPredictor().fit(X).predict_proba(X)[:, 1]
        assert (p > 0.99).all()

    def test_momentum_monotone(self):
        from src.v3.direction import MomentumPredictor
        X = pd.DataFrame({"mom_21": np.linspace(-0.2, 0.2, 50)})
        p = MomentumPredictor().fit(X).predict_proba(X)[:, 1]
        assert p[-1] > p[0]  # positive momentum → higher P(up)


class TestDirectionalMetrics:
    def test_perfect_predictor_high_accuracy(self):
        from src.v3.direction import directional_metrics
        rng = np.random.default_rng(0)
        y = (rng.random(400) > 0.5).astype(float)
        p = np.where(y == 1, 0.9, 0.1)  # perfect ranking
        m = directional_metrics(y, p)
        assert m["accuracy"] > 0.95
        assert m["edge_over_majority"] > 0

    def test_random_predictor_no_edge(self):
        from src.v3.direction import directional_metrics
        rng = np.random.default_rng(3)
        y = (rng.random(2000) > 0.45).astype(float)  # mild up-bias
        p = rng.random(2000)  # pure noise
        m = directional_metrics(y, p)
        # A noise predictor must NOT show a positive edge over the majority
        # baseline (small negative is expected; small positive from variance OK).
        assert m["edge_over_majority"] < 0.02
        assert abs(m["auc"] - 0.5) < 0.06  # AUC near chance

    def test_backtest_keys(self):
        from src.v3.direction import directional_backtest
        idx = pd.bdate_range("2010-01-01", periods=300)
        rng = np.random.default_rng(0)
        ret = pd.Series(rng.normal(0.0004, 0.012, 300), index=idx)
        prob = pd.Series(rng.random(300), index=idx)
        out = directional_backtest(ret, prob, mode="long_flat")
        assert {"strategy_sharpe", "buyhold_sharpe", "pct_time_long"} <= set(out)
