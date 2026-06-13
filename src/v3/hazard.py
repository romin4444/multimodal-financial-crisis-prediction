"""
v3 — Discrete-time hazard (survival) model for crisis onset.

WHY:
    Crisis onset is fundamentally time-to-event. Binary "crisis within N days"
    classification fudges two things: (a) right-edge censoring (the last N days
    have no resolvable label) and (b) the duration dependence of risk (how long
    since the last drawdown). A discrete-time hazard model handles both naturally
    and yields exactly the object a risk officer wants:

        P(a >= X% drawdown occurs within the next N days | information today).

HOW (dependency-light, the standard pooled-logistic discrete-time hazard):
    1. Identify drawdown ONSET days and "at-risk" days (not already in a drawdown).
    2. Fit logistic regression of onset ~ features + duration on at-risk days
       (this is the discrete hazard h_t = P(onset at t | survived to t)).
    3. Convert the daily hazard into an N-day cumulative incidence:
           P(event within N) = 1 - prod_{k=0..N-1} (1 - h_{t+k})
       For forecasting we use the model's hazard under current features as the
       per-step hazard (a standard, transparent approximation).
    4. CALIBRATE the N-day cumulative incidence with isotonic regression fit on
       a held-out slice of the training mask. This converts "ranks risk well"
       into "calibrated probability you can actually quote." (See v4 roadmap §2.1.)

CALIBRATION HISTORY (read this before re-introducing class_weight="balanced"):
    The original hazard model used class_weight="balanced". On a ~4% base rate
    that inflates every predicted probability ~20x — ranking (C-index) stays
    excellent, but the probabilities are useless (Brier skill ~ -2). v3.3 drops
    the re-weighting AND wraps the cumulative-incidence in isotonic calibration
    (the per-step hazard is already a reasonable Bernoulli; the compounded
    N-day risk is what most needs calibrating). The v3.1 fix on the fusion
    classifier was the precedent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def drawdown_panel(close: pd.Series, threshold: float = 0.10) -> pd.DataFrame:
    """
    Build the survival panel:
      - dd        : running drawdown from the trailing peak (<= 0).
      - in_dd     : currently in a >= `threshold` drawdown episode.
      - onset     : 1 on the day the drawdown first crosses below -threshold.
      - at_risk   : 1 on days NOT already inside a >= threshold drawdown.
      - duration  : trading days since the last episode ended (time at risk).
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    peak = np.maximum.accumulate(c)
    dd = c / peak - 1.0

    in_dd = dd <= -abs(threshold)
    onset = np.zeros(n, dtype=int)
    at_risk = np.zeros(n, dtype=int)
    duration = np.zeros(n, dtype=int)

    currently = False  # in a drawdown episode at the START of the day
    dur = 0
    for t in range(n):
        # Risk is assessed at the start of the day, BEFORE the event resolves,
        # so the onset day itself is an at-risk day (correct discrete-time
        # hazard convention — otherwise every positive is excluded).
        at_risk[t] = 0 if currently else 1
        if in_dd[t] and not currently:
            onset[t] = 1
            currently = True
        if not in_dd[t]:
            currently = False
        if at_risk[t]:
            dur += 1
        else:
            dur = 0
        duration[t] = dur

    return pd.DataFrame(
        {"dd": dd, "in_dd": in_dd.astype(int), "onset": onset,
         "at_risk": at_risk, "duration": duration},
        index=close.index,
    )


@dataclass
class HazardFit:
    model: object
    feature_cols: List[str]
    # v3.3: isotonic calibrator fit on a held-out training slice. Maps the
    # RAW cumulative incidence (1 - (1-h)^N) to a calibrated probability that
    # tracks the empirical N-day event frequency. None when fit_hazard was
    # called with calibrate=False or there were too few onsets to calibrate.
    incidence_calibrator: Optional[IsotonicRegression] = None
    calibrated_horizon: Optional[int] = None

    def hazard(self, X: pd.DataFrame) -> np.ndarray:
        """Per-day onset hazard h_t."""
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def cumulative_incidence(
        self, X: pd.DataFrame, horizon: int, calibrated: bool = True
    ) -> np.ndarray:
        """
        P(event within `horizon` days) ≈ 1 - (1 - h_t)^horizon, using the current
        per-day hazard as the constant-within-horizon rate (transparent approx).

        When `calibrated=True` (default) and an isotonic calibrator was fit at
        this horizon, the raw incidence is mapped through it — recommended for
        any downstream probability use.
        """
        h = np.clip(self.hazard(X), 1e-6, 1 - 1e-6)
        raw = 1.0 - (1.0 - h) ** horizon
        if (
            calibrated
            and self.incidence_calibrator is not None
            and self.calibrated_horizon == horizon
        ):
            return np.clip(self.incidence_calibrator.predict(raw), 1e-6, 1 - 1e-6)
        return raw


def fit_hazard(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    horizon: Optional[int] = None,
    calibrate: bool = True,
    calib_frac: float = 0.2,
    drawdown_threshold: float = 0.10,
) -> HazardFit:
    """
    Fit the pooled-logistic discrete-time hazard on AT-RISK training days only.

    Args:
        horizon: N-day horizon for the cumulative-incidence calibrator. Required
            when ``calibrate=True``.
        calibrate: If True, hold out the last ``calib_frac`` of the training mask
            (in time order), fit the LR on the rest, then fit an
            ``IsotonicRegression`` mapping raw N-day incidence to the realized
            N-day frequency on the held-out slice.
        drawdown_threshold: must match the threshold used to build ``panel``.

    Returns a ``HazardFit`` whose ``cumulative_incidence(X, horizon)`` returns
    calibrated probabilities when a calibrator was successfully fit.
    """
    cols = [c for c in feature_cols if c in features.columns]
    use_cols = cols + ["duration"]

    data = features.copy()
    data["duration"] = panel["duration"]
    data["onset"] = panel["onset"]
    data["at_risk"] = panel["at_risk"]
    data = data.dropna(subset=cols)

    tr_mask_series = (
        pd.Series(train_mask, index=features.index)
        .reindex(data.index)
        .fillna(False)
        .to_numpy()
    )
    tr_full = data[(data["at_risk"] == 1) & tr_mask_series]

    if calibrate and horizon is not None and len(tr_full) > 200:
        cut = int(len(tr_full) * (1.0 - calib_frac))
        tr_model = tr_full.iloc[:cut]
        tr_calib = tr_full.iloc[cut:]
    else:
        tr_model = tr_full
        tr_calib = tr_full.iloc[0:0]  # empty

    # v3.3: DROPPED class_weight="balanced". On a ~4% base rate it inflated
    # predicted probabilities ~20x, destroying Brier skill while leaving the
    # C-index intact (the v4-roadmap §2.1 finding). The un-weighted variant
    # produces probabilities the downstream calibrator can actually correct.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0),
    )
    model.fit(tr_model[use_cols], tr_model["onset"].astype(int))
    fit = HazardFit(model=model, feature_cols=use_cols)

    if calibrate and horizon is not None and len(tr_calib) > 50:
        realized_calib = _realized_within(
            panel, horizon, drawdown_threshold
        ).reindex(tr_calib.index)
        valid = realized_calib.notna().to_numpy()
        y_cal = realized_calib[valid].to_numpy().astype(int)
        if len(np.unique(y_cal)) > 1 and y_cal.sum() >= 3:
            raw_cal = fit.cumulative_incidence(
                tr_calib[valid], horizon, calibrated=False
            )
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_cal, y_cal)
            fit.incidence_calibrator = iso
            fit.calibrated_horizon = int(horizon)

    return fit


def evaluate_hazard(
    fit: HazardFit,
    panel: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int,
    test_mask: np.ndarray,
    drawdown_threshold: float = 0.10,
) -> dict:
    """
    Evaluate on at-risk TEST days:
      - C-index: AUC of the daily hazard score vs the observed onset (concordance).
      - N-day cumulative-incidence calibration vs the realized binary outcome
        (did a >= threshold drawdown actually occur within `horizon` days?).
      - Reports BOTH the raw and isotonic-calibrated N-day risk metrics so the
        calibration uplift is visible.
    """
    data = features.copy()
    data["duration"] = panel["duration"]
    data["onset"] = panel["onset"]
    data["at_risk"] = panel["at_risk"]
    data = data.dropna(subset=[c for c in fit.feature_cols if c != "duration"])

    mask = (data["at_risk"] == 1) & pd.Series(test_mask, index=features.index).reindex(data.index).fillna(False).to_numpy()
    te = data[mask]
    if len(te) < 50 or te["onset"].sum() < 3:
        return {"n": int(len(te)), "c_index": float("nan"), "note": "insufficient test onsets"}

    h = fit.hazard(te)
    c_index = float(roc_auc_score(te["onset"].astype(int), h)) if te["onset"].nunique() > 1 else float("nan")

    realized = _realized_within(panel, horizon, drawdown_threshold).reindex(te.index)
    valid = realized.notna().to_numpy()
    y = realized[valid].to_numpy().astype(int)

    raw_risk = fit.cumulative_incidence(te[valid], horizon, calibrated=False)
    cal_risk = fit.cumulative_incidence(te[valid], horizon, calibrated=True)
    raw_risk = np.clip(raw_risk, 1e-6, 1 - 1e-6)
    cal_risk = np.clip(cal_risk, 1e-6, 1 - 1e-6)

    def _scores(p):
        if len(np.unique(y)) <= 1:
            return float("nan"), float("nan")
        brier = float(brier_score_loss(y, p))
        base = float(np.mean(y))
        bss = float(1 - brier / brier_score_loss(y, np.full_like(p, base)))
        return brier, bss

    brier_raw, bss_raw = _scores(raw_risk)
    brier_cal, bss_cal = _scores(cal_risk)
    base_rate = float(np.mean(y)) if len(y) else float("nan")
    calibrated = fit.incidence_calibrator is not None

    return {
        "n": int(len(te)),
        "n_onsets": int(te["onset"].sum()),
        "c_index": round(c_index, 4),
        "horizon": horizon,
        "Nday_base_rate": round(base_rate, 4) if np.isfinite(base_rate) else None,
        # Raw N-day risk (1 - (1-h)^N, no calibration applied)
        "Nday_risk_brier_raw": round(brier_raw, 4) if np.isfinite(brier_raw) else None,
        "Nday_risk_brier_skill_raw": round(bss_raw, 4) if np.isfinite(bss_raw) else None,
        # Isotonic-calibrated N-day risk (the deployable number)
        "Nday_risk_brier_calibrated": round(brier_cal, 4) if np.isfinite(brier_cal) else None,
        "Nday_risk_brier_skill_calibrated": round(bss_cal, 4) if np.isfinite(bss_cal) else None,
        "calibrated": calibrated,
        # Back-compat aliases (older artifacts referenced these names):
        "Nday_risk_brier": round(brier_cal if calibrated else brier_raw, 4) if np.isfinite(brier_cal if calibrated else brier_raw) else None,
        "Nday_risk_brier_skill": round(bss_cal if calibrated else bss_raw, 4) if np.isfinite(bss_cal if calibrated else bss_raw) else None,
    }


def _realized_within(panel: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:  # noqa: ARG001 — threshold kept for API parity
    """For each at-risk day t, did a >= threshold drawdown onset occur in (t, t+horizon]?

    Vectorized: a rolling forward window sums the onset flag in (t, t+horizon].
    O(n) instead of the old O(n * horizon) Python loop.
    """
    onset = panel["onset"].to_numpy(dtype=float)
    n = len(onset)
    if n == 0:
        return pd.Series([], dtype=float, index=panel.index)

    # cumulative sum trick: events in (t, t+horizon] == cs[t+horizon+1] - cs[t+1]
    cs = np.concatenate([[0.0], np.cumsum(onset)])
    out = np.full(n, np.nan)
    for t in range(n - 1):
        j = min(t + 1 + horizon, n)  # exclusive upper bound on cs index
        out[t] = 1.0 if (cs[j] - cs[t + 1]) > 0 else 0.0
    return pd.Series(out, index=panel.index)
