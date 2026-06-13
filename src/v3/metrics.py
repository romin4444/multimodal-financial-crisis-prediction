"""
v3 — Honest metrics for rare-event probabilistic forecasting.

PROBLEM IN v2:
    Reported only F1 at a tuned threshold, on tiny all-positive windows where
    ROC-AUC was undefined (NaN). F1 on an all-positive set is meaningless.

FIX:
    Report metrics designed for imbalanced, probabilistic, real-time forecasting:
      - Precision-Recall AUC (average precision) — robust to imbalance.
      - Brier score + Brier Skill Score vs the base-rate (positive = real skill).
      - Expected Calibration Error — are the probabilities trustworthy?
      - Lift @ top-decile — operational value for an analyst triaging alerts.
      - Economic backtest — does acting on the signal improve Sharpe / drawdown?
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _clean(y_true: np.ndarray, y_prob: np.ndarray):
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    return y_true[mask].astype(int), np.clip(y_prob[mask], 1e-6, 1 - 1e-6)


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """BSS = 1 - Brier(model) / Brier(base-rate). >0 means better than climatology."""
    y, p = _clean(y_true, y_prob)
    if len(y) == 0:
        return float("nan")
    base = float(np.mean(y))
    bs_model = brier_score_loss(y, p)
    bs_base = brier_score_loss(y, np.full_like(p, base))
    if bs_base == 0:
        return float("nan")
    return float(1 - bs_model / bs_base)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Mean |predicted prob − observed frequency| across probability bins."""
    y, p = _clean(y_true, y_prob)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        conf = p[sel].mean()
        acc = y[sel].mean()
        ece += (sel.sum() / len(y)) * abs(conf - acc)
    return float(ece)


def lift_at_top_decile(y_true: np.ndarray, y_prob: np.ndarray, frac: float = 0.10) -> float:
    """How many more positives the top-`frac` riskiest days catch vs random."""
    y, p = _clean(y_true, y_prob)
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    k = max(1, int(len(p) * frac))
    top = np.argsort(p)[::-1][:k]
    precision_top = y[top].mean()
    base = y.mean()
    return float(precision_top / base) if base > 0 else float("nan")


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y, p = _clean(y_true, y_prob)
    out: Dict[str, float] = {}
    if len(y) == 0:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"),
                "brier": float("nan"), "brier_skill": float("nan"),
                "ece": float("nan"), "lift_top_decile": float("nan"),
                "base_rate": float("nan"), "n": 0}
    out["n"] = int(len(y))
    out["base_rate"] = round(float(y.mean()), 4)
    out["pr_auc"] = round(float(average_precision_score(y, p)), 4) if y.sum() > 0 else float("nan")
    out["roc_auc"] = round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else float("nan")
    out["brier"] = round(float(brier_score_loss(y, p)), 4)
    out["brier_skill"] = round(brier_skill_score(y, p), 4)
    out["ece"] = round(expected_calibration_error(y, p), 4)
    out["lift_top_decile"] = round(lift_at_top_decile(y, p), 4)
    return out


def economic_backtest(
    forward_returns: pd.Series,
    crisis_prob: pd.Series,
    threshold: float | None = 0.5,
    ann_factor: int = 252,
    quantile: float | None = None,
    quantile_warmup: int = 252,
) -> Dict[str, float]:
    """
    De-risking strategy: hold the market when P(crisis) < threshold, else go to
    cash. Compares to buy-and-hold on the same series.

    `forward_returns[t]` is the simple return earned from t to t+1 (next-day),
    so the decision at t uses only information available at t.

    If `quantile` is given (e.g. 0.85), the threshold at each day t is the
    `quantile` of crisis_prob over [:t] — an expanding, deployable threshold
    that uses only past predictions. During the `quantile_warmup` window the
    strategy stays fully in the market (no signal yet). This avoids the look-
    ahead bug of choosing the threshold from the full out-of-sample series.
    """
    idx = forward_returns.index.intersection(crisis_prob.index)
    r = forward_returns.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    p = crisis_prob.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    if quantile is not None:
        # Expanding quantile of past predictions: thr[t] = quantile of p[:t].
        # Shift by 1 so day t uses information strictly before t.
        p_series = pd.Series(p, index=idx)
        thr_series = p_series.expanding(min_periods=quantile_warmup).quantile(quantile).shift(1)
        thr_arr = thr_series.to_numpy(dtype=float)
        in_market = np.where(np.isnan(thr_arr), 1.0, (p < thr_arr).astype(float))
    else:
        thr = threshold if threshold is not None else 0.5
        in_market = (p < thr).astype(float)
    strat_r = in_market * r

    def _sharpe(x: np.ndarray) -> float:
        sd = x.std()
        return float(np.sqrt(ann_factor) * x.mean() / sd) if sd > 0 else float("nan")

    def _max_dd(x: np.ndarray) -> float:
        eq = np.cumprod(1 + x)
        peak = np.maximum.accumulate(eq)
        return float(((eq - peak) / peak).min())

    return {
        "buyhold_sharpe": round(_sharpe(r), 3),
        "strategy_sharpe": round(_sharpe(strat_r), 3),
        "buyhold_max_drawdown": round(_max_dd(r), 4),
        "strategy_max_drawdown": round(_max_dd(strat_r), 4),
        "buyhold_total_return": round(float(np.cumprod(1 + r)[-1] - 1), 4),
        "strategy_total_return": round(float(np.cumprod(1 + strat_r)[-1] - 1), 4),
        "pct_time_in_market": round(float(in_market.mean()), 4),
    }
