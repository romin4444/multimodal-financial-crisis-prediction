"""
Unit tests for v3 leakage-free modules.
Key tests assert the ABSENCE of look-ahead bias — the whole point of v3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TestLabeling:
    def test_forward_drawdown_is_causal_in_label_only(self):
        from src.v3.labeling import forward_max_drawdown
        # Monotonic up series → no forward drawdown
        close = pd.Series(np.arange(1, 101, dtype=float),
                          index=pd.bdate_range("2010-01-01", periods=100))
        fdd = forward_max_drawdown(close, horizon=5)
        # rising market: future min > current is false; drawdown ~ small/zero
        assert (fdd.dropna() >= -1e-9).all()

    def test_crash_is_labeled(self):
        from src.v3.labeling import crisis_label
        idx = pd.bdate_range("2010-01-01", periods=60)
        prices = np.concatenate([np.full(30, 100.0), np.linspace(100, 70, 30)])
        close = pd.Series(prices, index=idx)
        label = crisis_label(close, horizon=21, drawdown_threshold=0.10)
        # Days just before the crash should be labeled 1
        assert label.iloc[20:28].max() == 1.0

    def test_right_edge_is_nan_not_zero(self):
        from src.v3.labeling import crisis_label
        idx = pd.bdate_range("2010-01-01", periods=30)
        close = pd.Series(np.full(30, 100.0), index=idx)
        label = crisis_label(close, horizon=21, drawdown_threshold=0.10)
        # last row cannot have a full forward window → must be NaN, never fabricated 0
        assert np.isnan(label.iloc[-1])


class TestCausalRegime:
    def test_filtered_proba_sums_to_one(self):
        from src.v3.causal_regime import filtered_state_proba
        from hmmlearn.hmm import GaussianHMM
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 2))
        m = GaussianHMM(n_components=3, covariance_type="full", n_iter=20, random_state=0)
        m.fit(X)
        filt = filtered_state_proba(m, X)
        assert filt.shape == (300, 3)
        assert np.allclose(filt.sum(axis=1), 1.0, atol=1e-6)

    def test_filtered_differs_from_smoothed(self):
        """Filtered (causal) posteriors must NOT equal smoothed ones — proves
        we removed the future-information leak that v2 had. Uses OVERLAPPING
        regimes so there is genuine state uncertainty for smoothing to resolve."""
        from src.v3.causal_regime import filtered_state_proba
        from hmmlearn.hmm import GaussianHMM
        rng = np.random.default_rng(1)
        # Heavily overlapping regimes (separation ~1 sigma) + frequent switching
        states = (rng.random(400) < 0.5).astype(int)
        X = (rng.normal(0, 1, (400, 2)) + states[:, None] * 1.0)
        m = GaussianHMM(n_components=2, covariance_type="full", n_iter=50, random_state=1)
        m.fit(X)
        filtered = filtered_state_proba(m, X)
        smoothed = m.predict_proba(X)
        # Mean absolute difference should be materially non-zero
        assert np.abs(filtered - smoothed).mean() > 1e-3

    def test_filtered_is_truly_causal(self):
        """Filtered posterior at time t must be invariant to data AFTER t."""
        from src.v3.causal_regime import filtered_state_proba
        from hmmlearn.hmm import GaussianHMM
        rng = np.random.default_rng(2)
        X = rng.standard_normal((200, 2))
        m = GaussianHMM(n_components=3, covariance_type="full", n_iter=30, random_state=2)
        m.fit(X)
        full = filtered_state_proba(m, X)
        truncated = filtered_state_proba(m, X[:120])
        # First 120 filtered values identical whether or not future exists
        assert np.allclose(full[:120], truncated, atol=1e-8)


class TestBaselines:
    def test_base_rate_predicts_constant(self):
        from src.v3.baselines import BaseRatePredictor
        X = pd.DataFrame({"a": range(100)})
        y = np.array([1] * 30 + [0] * 70)
        p = BaseRatePredictor().fit(X, y).predict_proba(X)[:, 1]
        assert np.allclose(p, 0.30)

    def test_vix_threshold_monotone(self):
        from src.v3.baselines import VixThresholdPredictor
        X = pd.DataFrame({"vix": np.linspace(10, 80, 100)})
        m = VixThresholdPredictor().fit(X)
        p = m.predict_proba(X)[:, 1]
        assert p[-1] > p[0]  # higher VIX → higher crisis prob


class TestMetrics:
    def test_brier_skill_positive_for_good_model(self):
        from src.v3.metrics import brier_skill_score
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.2).astype(int)
        good = np.where(y == 1, 0.8, 0.1)  # informative
        assert brier_skill_score(y, good) > 0

    def test_brier_skill_zero_for_baserate(self):
        from src.v3.metrics import brier_skill_score
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.2).astype(int)
        base = np.full(500, y.mean())
        assert abs(brier_skill_score(y, base)) < 1e-6

    def test_economic_backtest_keys(self):
        from src.v3.metrics import economic_backtest
        idx = pd.bdate_range("2010-01-01", periods=200)
        rng = np.random.default_rng(0)
        ret = pd.Series(rng.normal(0.0003, 0.01, 200), index=idx)
        prob = pd.Series(rng.random(200), index=idx)
        out = economic_backtest(ret, prob, threshold=0.5)
        assert "strategy_sharpe" in out and "buyhold_max_drawdown" in out


class TestWalkForward:
    def test_no_lookahead_in_oos(self):
        """OOS predictions must only exist after min_train + embargo."""
        from src.v3.walkforward import WalkForwardConfig, walk_forward_predict
        from src.v3.baselines import BaseRatePredictor
        idx = pd.bdate_range("2000-01-01", periods=2000)
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"f": rng.standard_normal(2000)}, index=idx)
        y = pd.Series((rng.random(2000) < 0.15).astype(float), index=idx)
        cfg_ = WalkForwardConfig(min_train=1260, step=63, embargo=21, horizon=21)
        res = walk_forward_predict(X, y, BaseRatePredictor, cfg_)
        first_pred = res.oos_proba.notna().idxmax()
        first_pos = idx.get_loc(first_pred)
        assert first_pos >= 1260 + 21  # nothing predicted before train+embargo
        assert res.n_folds > 0
