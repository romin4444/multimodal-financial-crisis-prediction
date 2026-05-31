"""
v3 — Stock price DIRECTION detection (up/down over the next N trading days).

This replaces the crisis-prediction target with a directional-movement target,
while keeping the v3 discipline:
  - EXOGENOUS target: sign of the forward N-day return (not a model's own label).
  - CAUSAL features only (everything known at time t).
  - Honest baselines: always-up, momentum, base-rate.
  - Walk-forward evaluation (see src/v3/walkforward.py) + proper metrics.

Academic context: short-horizon direction of liquid stocks is close to a
random walk (Fama 1970, EMH). The honest bar is therefore: can a model beat the
"always predict up" majority baseline out-of-sample? We measure exactly that.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from src.v3.metrics import brier_skill_score, expected_calibration_error


# ─── Target ──────────────────────────────────────────────────────────────────

def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Simple return from t to t+horizon. NaN near the right edge."""
    return (close.shift(-horizon) / close - 1.0).rename("fwd_return")


def direction_label(close: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Binary up/down target: 1 if the stock is higher in `horizon` trading days.
    NaN at the right edge (no fabricated labels). Uses ONLY future prices to
    define the label — features must use only the past.
    """
    fr = forward_return(close, horizon)
    label = (fr > 0).astype("float")
    label[fr.isna()] = np.nan
    return label.rename("direction")


def direction_summary(label: pd.Series) -> dict:
    valid = label.dropna()
    return {
        "n": int(len(valid)),
        "n_up": int(valid.sum()),
        "up_rate": round(float(valid.mean()), 4) if len(valid) else None,
    }


# ─── Features ────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{n}")


def build_stock_features(stk: pd.DataFrame, market_ctx: pd.DataFrame) -> pd.DataFrame:
    """
    Per-stock technical features (all backward-looking) + merged market context.

    Args:
        stk: OHLCV DataFrame for one ticker (needs 'Close', 'Volume').
        market_ctx: DataFrame indexed on trading days with market-wide features
                    (e.g. vix, FSI, causal regime probs, market momentum).
    """
    close = stk["Close"]
    volume = stk.get("Volume", pd.Series(np.nan, index=stk.index))
    log_ret = np.log(close / close.shift(1))

    df = pd.DataFrame(index=stk.index)
    df["ret_1"] = close.pct_change()
    df["mom_5"] = close.pct_change(5)
    df["mom_21"] = close.pct_change(21)
    df["mom_63"] = close.pct_change(63)
    df["vol_21"] = log_ret.rolling(21).std() * np.sqrt(252)
    df["vol_63"] = log_ret.rolling(63).std() * np.sqrt(252)
    df["dist_ma50"] = close / close.rolling(50).mean() - 1
    df["dist_ma200"] = close / close.rolling(200).mean() - 1
    df["rsi_14"] = _rsi(close, 14)
    df["vol_ratio"] = volume / volume.rolling(21).mean()
    df["drawdown_63"] = close.rolling(63).apply(
        lambda x: (x[-1] - x.max()) / x.max() if x.max() != 0 else 0.0, raw=True
    )

    # Merge market-wide context (already causal)
    df = df.join(market_ctx.reindex(df.index), how="left")
    return df


# Feature groups for ablation
STOCK_FEATURE_COLS = [
    "ret_1", "mom_5", "mom_21", "mom_63", "vol_21", "vol_63",
    "dist_ma50", "dist_ma200", "rsi_14", "vol_ratio", "drawdown_63",
]
MARKET_CONTEXT_COLS = ["mkt_vix", "mkt_fsi", "mkt_prob_volatile", "mkt_prob_crisis", "mkt_mom_21"]


# ─── Directional baselines ────────────────────────────────────────────────────

class AlwaysUpPredictor:
    """Predict 'up' every day (P=1). Accuracy = test up-rate. The bar to beat."""

    def fit(self, X, y=None) -> "AlwaysUpPredictor":
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.full(len(X), 1.0 - 1e-6)
        return np.column_stack([1 - p, p])


class MomentumPredictor:
    """
    Classic technical baseline: predict 'up' when recent momentum is positive.
    Probability is a squashed function of the momentum feature.
    """

    def __init__(self, feature: str = "mom_21") -> None:
        self.feature = feature
        self._scale = 1.0

    def fit(self, X: pd.DataFrame, y=None) -> "MomentumPredictor":
        s = X[self.feature].std()
        self._scale = float(s) if s and np.isfinite(s) and s > 0 else 1.0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        m = X[self.feature].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-m / (self._scale + 1e-9)))  # logistic squash
        p = np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


# ─── Metrics ─────────────────────────────────────────────────────────────────

def directional_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Accuracy / AUC / MCC for a (near-balanced) direction task, plus the
    all-important EDGE over the always-predict-majority baseline."""
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y = y_true[mask].astype(int)
    p = np.clip(y_prob[mask], 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {"n": 0, "accuracy": float("nan"), "auc": float("nan"),
                "mcc": float("nan"), "majority_acc": float("nan"),
                "edge_over_majority": float("nan"), "brier_skill": float("nan"),
                "ece": float("nan"), "up_rate": float("nan")}
    pred = (p >= 0.5).astype(int)
    up_rate = float(y.mean())
    majority_acc = max(up_rate, 1 - up_rate)
    acc = float((pred == y).mean())
    return {
        "n": int(len(y)),
        "up_rate": round(up_rate, 4),
        "accuracy": round(acc, 4),
        "majority_acc": round(majority_acc, 4),
        "edge_over_majority": round(acc - majority_acc, 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else float("nan"),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4) if len(np.unique(pred)) > 1 else 0.0,
        "brier_skill": round(brier_skill_score(y.astype(float), p), 4),
        "ece": round(expected_calibration_error(y.astype(float), p), 4),
    }


def directional_backtest(
    fwd_returns: pd.Series,
    prob_up: pd.Series,
    mode: str = "long_flat",
    ann_factor: int = 252,
) -> Dict[str, float]:
    """
    Trade on the directional signal:
      - long_flat : long when P(up) >= 0.5, else cash.
      - long_short: long when P(up) >= 0.5, else short.
    Compares to buy-and-hold on the same series. `fwd_returns[t]` is the t->t+1
    return, so the decision at t uses only information available at t.
    """
    idx = fwd_returns.index.intersection(prob_up.index)
    r = fwd_returns.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    p = prob_up.reindex(idx).fillna(0.5).to_numpy(dtype=float)

    if mode == "long_short":
        pos = np.where(p >= 0.5, 1.0, -1.0)
    else:
        pos = np.where(p >= 0.5, 1.0, 0.0)
    strat = pos * r

    def _sharpe(x):
        sd = x.std()
        return float(np.sqrt(ann_factor) * x.mean() / sd) if sd > 0 else float("nan")

    def _maxdd(x):
        eq = np.cumprod(1 + x)
        peak = np.maximum.accumulate(eq)
        return float(((eq - peak) / peak).min())

    return {
        "mode": mode,
        "buyhold_sharpe": round(_sharpe(r), 3),
        "strategy_sharpe": round(_sharpe(strat), 3),
        "buyhold_max_drawdown": round(_maxdd(r), 4),
        "strategy_max_drawdown": round(_maxdd(strat), 4),
        "buyhold_total_return": round(float(np.cumprod(1 + r)[-1] - 1), 4),
        "strategy_total_return": round(float(np.cumprod(1 + strat)[-1] - 1), 4),
        "pct_time_long": round(float((pos > 0).mean()), 4),
    }
