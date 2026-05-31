"""
v3 — Probability calibration for the walk-forward harness.

PROBLEM:
    In the uncalibrated v3 runs every Brier Skill Score is negative — the output
    "probabilities" are not trustworthy (a 0.7 does not mean a 70% chance). This
    is fatal for a risk tool, where the probability IS the product.

FIX:
    Wrap any base estimator so that, inside each walk-forward training window, it:
      1. fits the base model on the EARLIER part of the window, then
      2. calibrates it (isotonic or Platt/sigmoid) on a held-out LATER slice of
         the SAME window — never touching the test fold.
    This respects time order (no shuffled CV across time) and turns the API
    output from decorative into usable. Target: ECE < 0.03, Brier Skill > 0.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratedTimeSeriesClassifier:
    """
    Time-aware calibration wrapper. `base_factory()` returns a fresh, unfitted
    sklearn-compatible estimator each call.

    Calibration is done DIRECTLY on the base model's predicted probabilities
    over a held-out later slice of the training window — isotonic regression
    (or Platt/logistic when positives are sparse). This avoids
    CalibratedClassifierCV's cross-version `cv="prefit"` behaviour drift and is
    fully transparent and deterministic across scikit-learn versions.
    """

    def __init__(
        self,
        base_factory: Callable[[], object],
        method: str = "isotonic",
        calib_frac: float = 0.2,
        min_calib: int = 150,
        min_calib_pos: int = 8,
    ) -> None:
        self.base_factory = base_factory
        self.method = method
        self.calib_frac = calib_frac
        self.min_calib = min_calib
        self.min_calib_pos = min_calib_pos
        self.base_ = None
        self.calibrator_ = None
        self.method_ = None
        self.calibrated_ = False

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        n = len(y)
        cut = int(n * (1 - self.calib_frac))

        if hasattr(X, "iloc"):
            Xtr, Xcal = X.iloc[:cut], X.iloc[cut:]
        else:
            Xtr, Xcal = X[:cut], X[cut:]
        ytr, ycal = y[:cut], y[cut:]

        # Too small / single-class calibration slice → skip calibration cleanly
        if (len(ycal) < self.min_calib or len(np.unique(ycal)) < 2
                or ycal.sum() < self.min_calib_pos):
            self.base_ = self.base_factory().fit(X, y)
            self.calibrated_ = False
            return self

        self.base_ = self.base_factory().fit(Xtr, ytr)
        p_cal = self._base_proba(Xcal)

        method = self.method
        if method == "isotonic" and ycal.sum() < 25:
            method = "sigmoid"  # isotonic is unstable with very few positives

        try:
            if method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                cal.fit(p_cal, ycal)
            else:  # Platt scaling: 1-D logistic on the base probability
                cal = LogisticRegression(max_iter=1000)
                cal.fit(p_cal.reshape(-1, 1), ycal)
            self.calibrator_ = cal
            self.method_ = method
            self.calibrated_ = True
        except Exception:
            self.base_ = self.base_factory().fit(X, y)
            self.calibrated_ = False
        return self

    def _base_proba(self, X) -> np.ndarray:
        return self.base_.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        p = self._base_proba(X)
        if self.calibrated_:
            if self.method_ == "isotonic":
                pc = self.calibrator_.predict(p)
            else:
                pc = self.calibrator_.predict_proba(p.reshape(-1, 1))[:, 1]
            p = np.clip(np.nan_to_num(pc, nan=p), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_calibrated_factory(
    base_factory: Callable[[], object],
    method: str = "isotonic",
    calib_frac: float = 0.2,
) -> Callable[[], CalibratedTimeSeriesClassifier]:
    """Return a model_factory (zero-arg callable) producing calibrated wrappers,
    ready to drop into walk_forward_predict()."""
    def factory():
        return CalibratedTimeSeriesClassifier(base_factory, method=method, calib_frac=calib_frac)
    return factory
