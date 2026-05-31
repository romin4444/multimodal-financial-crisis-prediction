"""
v3 — Naive baselines.

PROBLEM IN v2:
    F1 = 0.99 was reported with NO baseline. A number is only meaningful relative
    to the simplest thing that could possibly work. On heavily imbalanced crisis
    windows, even "always predict crisis" scores a high F1.

FIX:
    Every model is benchmarked against:
      1. Base-rate (always predict the unconditional positive rate).
      2. VIX threshold (the industry-standard fear gauge as a one-feature rule).
      3. Persistence (recent realized drawdown predicts forward drawdown).
    A model earns its keep only by beating ALL of these out-of-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BaseRatePredictor:
    """Predicts the constant training-set positive rate for every sample."""

    def __init__(self) -> None:
        self.rate_ = 0.0

    def fit(self, X, y) -> "BaseRatePredictor":
        self.rate_ = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


class VixThresholdPredictor:
    """
    One-feature baseline: maps VIX to a crisis probability via the empirical CDF
    of VIX on the training set. High VIX → high probability. No ML.
    """

    def __init__(self, vix_col: str = "vix") -> None:
        self.vix_col = vix_col
        self._sorted_train = None

    def fit(self, X: pd.DataFrame, y=None) -> "VixThresholdPredictor":
        self._sorted_train = np.sort(X[self.vix_col].to_numpy(dtype=float))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        vix = X[self.vix_col].to_numpy(dtype=float)
        # empirical CDF position = P(VIX_train <= vix)
        p = np.searchsorted(self._sorted_train, vix, side="right") / max(
            len(self._sorted_train), 1
        )
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


class PersistencePredictor:
    """
    Recent realized drawdown predicts forward drawdown. Maps a 'badness' feature
    (e.g. drawdown_63, more negative = worse) to probability via training CDF.
    """

    def __init__(self, feature: str = "drawdown_63") -> None:
        self.feature = feature
        self._sorted_train = None

    def fit(self, X: pd.DataFrame, y=None) -> "PersistencePredictor":
        # badness = -drawdown (positive when in a drawdown)
        self._sorted_train = np.sort(-X[self.feature].to_numpy(dtype=float))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        badness = -X[self.feature].to_numpy(dtype=float)
        p = np.searchsorted(self._sorted_train, badness, side="right") / max(
            len(self._sorted_train), 1
        )
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])
