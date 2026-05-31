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

This complements the binary classifier; we report concordance (C-index = AUC of
the hazard score against onset) and calibration of the N-day risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
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

    def hazard(self, X: pd.DataFrame) -> np.ndarray:
        """Per-day onset hazard h_t."""
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def cumulative_incidence(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        P(event within `horizon` days) ≈ 1 - (1 - h_t)^horizon, using the current
        per-day hazard as the constant-within-horizon rate (transparent approx).
        """
        h = np.clip(self.hazard(X), 1e-6, 1 - 1e-6)
        return 1.0 - (1.0 - h) ** horizon


def fit_hazard(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
) -> HazardFit:
    """
    Fit the pooled-logistic discrete-time hazard on AT-RISK training days only.
    `duration` is included as a covariate to absorb baseline duration dependence.
    """
    cols = [c for c in feature_cols if c in features.columns]
    use_cols = cols + ["duration"]

    data = features.copy()
    data["duration"] = panel["duration"]
    data["onset"] = panel["onset"]
    data["at_risk"] = panel["at_risk"]

    data = data.dropna(subset=cols)
    # restrict to at-risk training days
    tr = data[(data["at_risk"] == 1) & pd.Series(train_mask, index=features.index).reindex(data.index).fillna(False).to_numpy()]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0),
    )
    model.fit(tr[use_cols], tr["onset"].astype(int))
    return HazardFit(model=model, feature_cols=use_cols)


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

    # Realized N-day outcome: did a >=threshold drawdown occur within horizon?
    close = panel["dd"]  # not used directly; recompute realized from dd path
    realized = _realized_within(panel, horizon, drawdown_threshold).reindex(te.index)
    risk = fit.cumulative_incidence(te, horizon)

    valid = realized.notna().to_numpy()
    y = realized[valid].to_numpy().astype(int)
    p = np.clip(risk[valid], 1e-6, 1 - 1e-6)

    brier = float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan")
    base = float(np.mean(y)) if len(y) else float("nan")
    bss = float(1 - brier / brier_score_loss(y, np.full_like(p, base))) if len(np.unique(y)) > 1 else float("nan")

    return {
        "n": int(len(te)),
        "n_onsets": int(te["onset"].sum()),
        "c_index": round(c_index, 4),
        "horizon": horizon,
        "Nday_risk_brier": round(brier, 4) if np.isfinite(brier) else None,
        "Nday_risk_brier_skill": round(bss, 4) if np.isfinite(bss) else None,
        "Nday_base_rate": round(base, 4) if np.isfinite(base) else None,
    }


def _realized_within(panel: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """For each at-risk day t, did a >= threshold drawdown onset occur in (t, t+horizon]?"""
    onset = panel["onset"].to_numpy()
    n = len(onset)
    out = np.full(n, np.nan)
    for t in range(n - 1):
        j = min(t + horizon, n - 1)
        out[t] = 1.0 if onset[t + 1 : j + 1].any() else 0.0
    return pd.Series(out, index=panel.index)
