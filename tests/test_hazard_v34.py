"""Smoke + regression tests for the v3.4 hazard calibrator.

The v3.3 calibrator fit isotonic on the temporal tail of the train window — on
1990->2010 splits the tail is the GFC, so the calibrator over-predicted on the
calmer post-2010 test set and turned the calibrated Brier skill NEGATIVE. v3.4:
sigmoid (Platt) on time-series OOF + do-no-harm guard (falls back to raw if it
can't confirm a benefit on the most-recent OOF slice).

These tests pin the contract: (a) the fit object always exposes a
``calib_method`` string; (b) "identity" fallback returns RAW for the cumulative
incidence; (c) a sigmoid calibrator is strictly monotone (cannot invert the
hazard ranking).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v3.hazard import (
    HazardFit,
    _IdentityCalibrator,
    _SigmoidCalibrator,
    drawdown_panel,
    evaluate_hazard,
    fit_hazard,
)


def _toy_panel(seed: int = 0, n: int = 1500):
    """A synthetic series with two ~25% drawdowns — enough onsets to fit."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    r = rng.normal(0.0004, 0.01, n)
    # Two engineered drawdowns at known windows so onsets are stable across runs.
    for cs, s in [(400, 0.04), (1000, 0.05)]:
        ce = cs + 60
        r[cs:ce] += rng.normal(-0.002, s, ce - cs)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx, name="close")
    feats = pd.DataFrame({
        "vol_21d": pd.Series(r, index=idx).rolling(21).std() * np.sqrt(252),
        "mom_21d": pd.Series(r, index=idx).rolling(21).mean() * 252,
        "drawdown_63": (close / close.cummax() - 1).rolling(63).min(),
    }, index=idx)
    panel = drawdown_panel(close, threshold=0.10)
    return panel, feats, close


class TestSigmoidCalibrator:
    def test_monotone_when_slope_positive(self):
        cal = _SigmoidCalibrator()
        raw = np.linspace(0.01, 0.99, 50)
        # Synthetic monotone-positive label so slope > 0 is guaranteed.
        y = (raw + np.random.default_rng(0).normal(0, 0.05, 50) > 0.5).astype(int)
        if y.sum() < 2 or y.sum() == len(y):
            pytest.skip("synthetic labels degenerate")
        cal.fit(raw, y)
        out = cal.predict(raw)
        assert cal.ok, "positive correlation should yield slope > 0"
        # Monotone increasing on the sorted input -> never inverts ranking.
        assert np.all(np.diff(out) >= -1e-12)


class TestIdentityCalibrator:
    def test_passthrough(self):
        cal = _IdentityCalibrator()
        x = np.array([-0.1, 0.0, 0.3, 1.2])
        # Clipped to [0,1], otherwise identical.
        assert np.allclose(cal.predict(x), np.array([0.0, 0.0, 0.3, 1.0]))


class TestFitHazardEndToEnd:
    """Full fit→evaluate flow on a toy panel. The default v3.4 method
    ("sigmoid_oof") must always produce a HazardFit whose calibrated incidence
    is no worse than raw on the test slice (do-no-harm guarantee)."""

    def test_v34_do_no_harm(self):
        panel, feats, _ = _toy_panel(seed=1, n=1800)
        if panel["onset"].sum() < 5:
            pytest.skip("not enough engineered onsets")
        n = len(feats)
        tr_mask = np.zeros(n, dtype=bool)
        tr_mask[: int(n * 0.6)] = True

        fit = fit_hazard(
            panel, feats, ["vol_21d", "mom_21d", "drawdown_63"], tr_mask,
            horizon=21, calibrate=True, calib_method="sigmoid_oof",
        )
        assert isinstance(fit, HazardFit)
        # Every fit advertises which calibrator survived the guard.
        assert fit.calib_method in {"sigmoid_oof", "isotonic_oof", "identity"}

        te_mask = ~tr_mask
        out = evaluate_hazard(fit, panel, feats, horizon=21, test_mask=te_mask)
        if "Nday_risk_brier_skill_raw" in out and out["Nday_risk_brier_skill_raw"] is not None:
            raw = out["Nday_risk_brier_skill_raw"]
            cal = out["Nday_risk_brier_skill_calibrated"]
            # Do-no-harm guarantee: calibrated within 1pp of raw (tolerance for
            # the toy panel's small-n noise).
            assert cal is None or cal >= raw - 0.01, (
                f"v3.4 calibration regressed vs raw: raw={raw} cal={cal} "
                f"method={fit.calib_method}"
            )
