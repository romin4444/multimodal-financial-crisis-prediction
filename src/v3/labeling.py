"""
v3 — Exogenous crisis labeling.

PROBLEM IN v2:
    The fusion target was `(HMM_regime == Crisis)` shifted forward. But the
    HMM "Crisis" state is an UNSUPERVISED cluster defined by high volatility,
    and the fusion features themselves include FSI / vol_21d / vix / prob_crisis.
    So the model was predicting a label derived from its own inputs — the
    near-perfect F1 (0.99) is volatility persistence, not crisis prediction.

FIX:
    Define the target from a FORWARD, EXOGENOUS, TRADABLE market outcome that
    is independent of any model: the realized peak-to-trough drawdown over the
    next `horizon` trading days. This is ground truth a risk officer cares about,
    and it cannot be trivially reconstructed from contemporaneous volatility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    """
    For each day t, the worst (most negative) return achievable by buying at
    close[t] and selling at the minimum close over (t, t+horizon].

    Returns a Series of forward drawdowns in [-1, +inf); NaN near the right edge.
    Uses ONLY future prices to define the label — features must use only past.
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n - 1):
        j = min(i + horizon, n - 1)
        future = c[i + 1 : j + 1]
        if future.size:
            out[i] = (future.min() - c[i]) / c[i]
    return pd.Series(out, index=close.index, name="fwd_drawdown")


def crisis_label(
    close: pd.Series,
    horizon: int = 21,
    drawdown_threshold: float = 0.10,
) -> pd.Series:
    """
    Binary early-warning target: will the market fall more than
    `drawdown_threshold` (e.g. 10%) within the next `horizon` trading days?

    This is the exogenous ground truth that breaks the v2 circularity.
    """
    fdd = forward_max_drawdown(close, horizon)
    label = (fdd <= -abs(drawdown_threshold)).astype("float")
    label[fdd.isna()] = np.nan  # don't fabricate labels at the right edge
    return label.rename("crisis_label")


def label_summary(label: pd.Series) -> dict:
    valid = label.dropna()
    return {
        "n": int(len(valid)),
        "n_positive": int(valid.sum()),
        "base_rate": round(float(valid.mean()), 4) if len(valid) else None,
    }
