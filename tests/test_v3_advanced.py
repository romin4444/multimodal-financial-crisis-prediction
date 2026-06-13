"""
Tests for the advanced v3 modules: calibration, macro features, online regime,
and the discrete-time hazard model. Synthetic data only — no network/GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Calibration ─────────────────────────────────────────────────────────────

class TestCalibration:
    def _data(self, n=1500, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
        logit = 1.5 * X["f1"] - 1.0 * X["f2"]
        p = 1 / (1 + np.exp(-logit))
        y = (rng.random(n) < p).astype(int)
        return X, pd.Series(y)

    def test_probabilities_in_range(self):
        from src.v3.calibration import CalibratedTimeSeriesClassifier
        from sklearn.linear_model import LogisticRegression
        X, y = self._data()
        clf = CalibratedTimeSeriesClassifier(lambda: LogisticRegression(max_iter=500))
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()

    def test_calibrates_when_enough_data(self):
        from src.v3.calibration import CalibratedTimeSeriesClassifier
        from sklearn.linear_model import LogisticRegression
        X, y = self._data(n=2000)
        clf = CalibratedTimeSeriesClassifier(lambda: LogisticRegression(max_iter=500)).fit(X, y)
        assert clf.calibrated_ is True

    def test_falls_back_on_single_class_calib(self):
        from src.v3.calibration import CalibratedTimeSeriesClassifier
        from sklearn.linear_model import LogisticRegression
        # Construct so the LATER (calibration) slice is all one class
        n = 600
        X = pd.DataFrame({"f1": np.arange(n, dtype=float)})
        y = pd.Series(np.r_[np.ones(300, dtype=int), np.zeros(300, dtype=int)])
        clf = CalibratedTimeSeriesClassifier(lambda: LogisticRegression(max_iter=500)).fit(X, y)
        assert clf.calibrated_ is False  # gracefully skipped
        p = clf.predict_proba(X)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()

    def test_improves_calibration_error(self):
        """Calibrated probabilities should have ECE no worse than uncalibrated."""
        from src.v3.calibration import CalibratedTimeSeriesClassifier
        from src.v3.metrics import expected_calibration_error
        from sklearn.ensemble import RandomForestClassifier
        X, y = self._data(n=2500, seed=5)
        cut = 1800
        base = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=0).fit(X[:cut], y[:cut])
        ece_base = expected_calibration_error(y[cut:].to_numpy(float), base.predict_proba(X[cut:])[:, 1])
        clf = CalibratedTimeSeriesClassifier(
            lambda: RandomForestClassifier(n_estimators=60, max_depth=4, random_state=0)
        ).fit(X[:cut], y[:cut])
        ece_cal = expected_calibration_error(y[cut:].to_numpy(float), clf.predict_proba(X[cut:])[:, 1])
        # allow small noise tolerance
        assert ece_cal <= ece_base + 0.02


# ─── Macro features ──────────────────────────────────────────────────────────

class TestMacroFeatures:
    def _market(self, n=400, seed=1):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2010-01-01", periods=n)
        out = {}
        for k in ["sp500", "vix", "aapl", "jpm", "xom", "gs"]:
            close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            out[k] = pd.DataFrame({"Close": close, "Volume": rng.integers(1e6, 5e6, n)}, index=idx)
        return out, idx

    def test_cross_asset_corr_bounded(self):
        from src.v3.macro_features import cross_asset_correlation
        market, _ = self._market()
        xc = cross_asset_correlation(market, window=63).dropna()
        assert (xc >= -1.0001).all() and (xc <= 1.0001).all()

    def test_build_macro_columns(self):
        from src.v3.macro_features import build_macro_features
        market, idx = self._market()
        rng = np.random.default_rng(2)
        fred = pd.DataFrame({
            "credit_spread": rng.uniform(3, 8, len(idx)),
            "yield_spread": rng.uniform(-1, 3, len(idx)),
            "fed_funds": rng.uniform(0, 5, len(idx)),
            "oil_price": rng.uniform(40, 100, len(idx)),
        }, index=idx)
        out = build_macro_features(fred, market, idx)
        assert "credit_spread" in out.columns
        assert "xasset_corr" in out.columns


# ─── Online regime ───────────────────────────────────────────────────────────

class TestOnlineRegime:
    def test_online_regime_shapes_and_probs(self):
        from src.v3.online_features import compute_online_regime
        rng = np.random.default_rng(3)
        n = 1600
        idx = pd.bdate_range("2005-01-01", periods=n)
        log_ret = rng.normal(0, 0.01, n)
        feat = pd.DataFrame({
            "log_ret": log_ret,
            "vol_21d": pd.Series(log_ret, index=idx).rolling(21).std().bfill().to_numpy() * np.sqrt(252),
            "FSI": np.clip(rng.random(n), 0, 1),
        }, index=idx)
        out = compute_online_regime(feat, ["log_ret", "vol_21d", "FSI"],
                                    min_train=600, refit_step=400, n_seeds=3, n_iter=40)
        prob_cols = ["o_prob_stable", "o_prob_volatile", "o_prob_crisis"]
        assert all(c in out.columns for c in prob_cols)
        sums = out[prob_cols].sum(axis=1).dropna()
        assert np.allclose(sums, 1.0, atol=1e-5)
        assert set(out["o_regime"].dropna().unique()).issubset({0, 1, 2})


# ─── Hazard ──────────────────────────────────────────────────────────────────

class TestHazard:
    def _crashy_close(self, seed=4):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2000-01-01", periods=1500)
        ret = rng.normal(0.0004, 0.01, 1500)
        # inject two crash episodes
        ret[500:530] = rng.normal(-0.02, 0.03, 30)
        ret[1000:1025] = rng.normal(-0.025, 0.03, 25)
        close = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
        return close

    def test_panel_detects_onsets(self):
        from src.v3.hazard import drawdown_panel
        panel = drawdown_panel(self._crashy_close(), threshold=0.10)
        assert panel["onset"].sum() >= 1
        assert set(panel["at_risk"].unique()).issubset({0, 1})
        assert (panel["duration"] >= 0).all()

    def test_fit_and_evaluate_runs(self):
        from src.v3.hazard import drawdown_panel, fit_hazard, evaluate_hazard
        close = self._crashy_close()
        idx = close.index
        feat = pd.DataFrame({
            "vol_21d": np.log(close / close.shift(1)).rolling(21).std().bfill() * np.sqrt(252),
            "drawdown_63": (close / close.rolling(63).max() - 1).bfill(),
            "mom_21d": close.pct_change(21).bfill(),
        }, index=idx)
        panel = drawdown_panel(close, threshold=0.10)
        cut = int(len(idx) * 0.6)
        tr = np.zeros(len(idx), dtype=bool)
        tr[:cut] = True
        fit = fit_hazard(panel, feat, ["vol_21d", "drawdown_63", "mom_21d"], tr)
        m = evaluate_hazard(fit, panel, feat, horizon=21, test_mask=~tr, drawdown_threshold=0.10)
        assert "c_index" in m and "n" in m

    def _many_crash_close(self, seed=11):
        """Longer series with several crash + RECOVERY episodes so both the
        model-train and calibration-holdout slices have multiple onsets to
        train on. Each crash drops the price ~15% over 20 days, then a
        recovery period puts a new trailing peak above the prior one so the
        next crash registers as a fresh onset."""
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("1990-01-01", periods=4000)
        ret = rng.normal(0.0006, 0.008, 4000)
        for start in (300, 700, 1200, 1700, 2300, 2900, 3400, 3700):
            # 20-day crash, then a strong recovery to make a new peak
            ret[start:start + 20] = rng.normal(-0.012, 0.02, 20)
            ret[start + 20:start + 120] = rng.normal(0.003, 0.008, 100)
        close = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
        return close

    def test_calibrated_incidence_is_within_unit_interval(self):
        """v3.3 — passing horizon=N to fit_hazard must attach a calibrator
        whose output stays in [0, 1] and whose ranking matches the raw."""
        from src.v3.hazard import drawdown_panel, fit_hazard
        close = self._many_crash_close()
        idx = close.index
        feat = pd.DataFrame({
            "vol_21d": np.log(close / close.shift(1)).rolling(21).std().bfill() * np.sqrt(252),
            "drawdown_63": (close / close.rolling(63).max() - 1).bfill(),
            "mom_21d": close.pct_change(21).bfill(),
        }, index=idx)
        panel = drawdown_panel(close, threshold=0.10)
        cut = int(len(idx) * 0.6)
        tr = np.zeros(len(idx), dtype=bool)
        tr[:cut] = True
        fit = fit_hazard(
            panel, feat, ["vol_21d", "drawdown_63", "mom_21d"], tr,
            horizon=21, calibrate=True, drawdown_threshold=0.10,
        )
        assert fit.incidence_calibrator is not None, (
            "horizon was supplied but no calibrator was attached"
        )
        assert fit.calibrated_horizon == 21
        # cumulative_incidence needs the "duration" covariate present —
        # join it from the panel before predicting.
        te = feat.iloc[cut:].copy()
        te["duration"] = panel["duration"].iloc[cut:]
        raw = fit.cumulative_incidence(te, horizon=21, calibrated=False)
        cal = fit.cumulative_incidence(te, horizon=21, calibrated=True)
        assert ((raw >= 0) & (raw <= 1)).all()
        assert ((cal >= 0) & (cal <= 1)).all()
        # Isotonic is non-decreasing by construction, but maps many distinct
        # inputs to the same output (step function) — collapsed ties drag
        # Kendall's tau below 1. The real invariant is: NO strict-order pair
        # gets inverted. Check the weaker rank-correlation threshold (catches
        # a destroyed ranking) plus the no-inversion invariant directly.
        from scipy.stats import kendalltau
        tau, _ = kendalltau(raw, cal)
        assert tau > 0.80, f"isotonic broke ranking (tau={tau:.3f})"
        # No inversion: every strict-order pair in raw must be non-strict in cal.
        order = np.argsort(raw)
        sorted_cal = cal[order]
        assert (np.diff(sorted_cal) >= -1e-9).all(), "isotonic produced an inversion"

    def test_no_class_weight_balanced_in_hazard_fit(self):
        """Regression guard: v3.3 explicitly dropped class_weight='balanced'
        from the hazard LR (the v4-roadmap §2.1 finding). If anyone copies
        it back in, this test must scream."""
        from src.v3.hazard import drawdown_panel, fit_hazard
        close = self._crashy_close()
        idx = close.index
        feat = pd.DataFrame({
            "vol_21d": np.log(close / close.shift(1)).rolling(21).std().bfill() * np.sqrt(252),
            "drawdown_63": (close / close.rolling(63).max() - 1).bfill(),
            "mom_21d": close.pct_change(21).bfill(),
        }, index=idx)
        panel = drawdown_panel(close, threshold=0.10)
        tr = np.ones(len(idx), dtype=bool)
        fit = fit_hazard(
            panel, feat, ["vol_21d", "drawdown_63", "mom_21d"], tr,
            calibrate=False,
        )
        lr = fit.model.named_steps["logisticregression"]
        # sklearn stores the resolved value; None == default ("not balanced").
        assert lr.class_weight is None, (
            "Hazard LR is using class_weight='balanced' again — this destroys "
            "calibration on the ~4% base rate. See v3.3 / v4-roadmap §2.1."
        )
