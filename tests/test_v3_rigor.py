"""
Tests for the frontier rigor modules: CPCV, PBO, Deflated Sharpe Ratio, and the
TDA feature builder (which must degrade gracefully when no TDA backend exists).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TestDeflatedSharpe:
    def test_psr_monotonic_in_sharpe(self):
        from src.v3.deflated_sharpe import probabilistic_sharpe_ratio
        lo = probabilistic_sharpe_ratio(0.02, 1000)
        hi = probabilistic_sharpe_ratio(0.10, 1000)
        assert hi > lo
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

    def test_more_trials_raise_benchmark(self):
        """More trials -> higher expected-max Sharpe (harder to be significant)."""
        from src.v3.deflated_sharpe import expected_max_sharpe
        rng = np.random.default_rng(0)
        sr = rng.normal(0, 0.05, 200)
        few = expected_max_sharpe(sr[:10])
        many = expected_max_sharpe(sr)
        assert many >= few

    def test_dsr_outputs_probability(self):
        from src.v3.deflated_sharpe import deflated_sharpe_ratio
        rng = np.random.default_rng(1)
        selected = rng.normal(0.0006, 0.01, 1500)   # mildly positive
        trials = rng.normal(0.0, 0.04, 30)
        out = deflated_sharpe_ratio(selected, trials)
        assert out["deflated_sharpe"] is None or 0.0 <= out["deflated_sharpe"] <= 1.0
        assert out["n_trials"] == 30

    def test_dsr_penalizes_pure_noise(self):
        """A zero-edge strategy among many trials should NOT look significant."""
        from src.v3.deflated_sharpe import deflated_sharpe_ratio
        rng = np.random.default_rng(2)
        selected = rng.normal(0.0, 0.01, 1500)  # no real edge
        trials = rng.normal(0.0, 0.05, 50)
        out = deflated_sharpe_ratio(selected, trials)
        assert out["deflated_sharpe"] < 0.95  # not significant


class TestCPCV:
    def _xy(self, n=1500, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2005-01-01", periods=n)
        X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)}, index=idx)
        signal = 1.2 * X["f1"]
        p = 1 / (1 + np.exp(-signal))
        y = pd.Series((rng.random(n) < p).astype(float), index=idx)
        return X, y

    def test_splits_no_overlap_and_embargo(self):
        from src.v3.cpcv import cpcv_splits, CPCVConfig
        n = 600
        cfg = CPCVConfig(n_groups=6, n_test_groups=2, embargo=10)
        seen = 0
        for train_idx, test_idx in cpcv_splits(n, cfg):
            # no overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            # embargo respected: no train index within `embargo` of the test span
            lo, hi = test_idx.min(), test_idx.max()
            band = set(range(max(0, lo - cfg.embargo), min(n, hi + cfg.embargo + 1)))
            assert band.isdisjoint(set(train_idx.tolist()))
            seen += 1
        assert seen == 15  # C(6, 2)

    def test_run_cpcv_produces_oos(self):
        from src.v3.cpcv import run_cpcv, CPCVConfig
        from sklearn.linear_model import LogisticRegression
        X, y = self._xy()
        res = run_cpcv(X, y, lambda: LogisticRegression(max_iter=500),
                       CPCVConfig(n_groups=6, n_test_groups=2, embargo=10))
        assert res.n_splits > 0
        assert res.oos_proba.notna().sum() > 0
        valid = res.oos_proba.dropna()
        assert ((valid >= 0) & (valid <= 1)).all()

    def test_pbo_in_unit_interval(self):
        from src.v3.cpcv import probability_of_backtest_overfitting
        rng = np.random.default_rng(3)
        # 12 time-slices x 8 strategies of pure noise -> PBO should be ~0.5
        perf = rng.normal(0.3, 0.05, size=(12, 8))
        out = probability_of_backtest_overfitting(perf, n_partitions=8)
        assert 0.0 <= out["pbo"] <= 1.0

    def test_pbo_low_for_genuinely_best_strategy(self):
        from src.v3.cpcv import probability_of_backtest_overfitting
        rng = np.random.default_rng(4)
        perf = rng.normal(0.2, 0.03, size=(12, 6))
        perf[:, 0] += 0.5  # strategy 0 is consistently, genuinely best
        out = probability_of_backtest_overfitting(perf, n_partitions=8)
        assert out["pbo"] < 0.5  # a real winner is rarely an overfit artefact


class TestTDA:
    def test_degrades_gracefully_without_backend(self):
        """build_tda_features must never crash; empty frame if no TDA lib."""
        from src.v3.tda_features import build_tda_features, tda_available
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2010-01-01", periods=300)
        rets = pd.Series(rng.normal(0, 0.01, 300), index=idx)
        out = build_tda_features(rets, window=63, embed_dim=3)
        # Either real features (if a backend is installed) or an empty-but-typed frame
        assert list(out.columns) == ["tda_total_persistence", "tda_max_persistence_h1", "tda_landscape_l2"]
        if tda_available():
            assert out.notna().any().any()
