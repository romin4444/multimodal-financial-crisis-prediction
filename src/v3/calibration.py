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
from sklearn.calibration import CalibratedClassifierCV


class CalibratedTimeSeriesClassifier:
    """
    Time-aware calibration wrapper. `base_factory()` returns a fresh, unfitted
    sklearn-compatible estimator each call.
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
        self.model_ = None
        self.calibrated_ = False

    def fit(self, X, y):
        X = np.asarray(X) if not hasattr(X, "iloc") else X
        y = np.asarray(y).astype(int)
        n = len(y)
        cut = int(n * (1 - self.calib_frac))

        # Split preserving time order
        if hasattr(X, "iloc"):
            Xtr, Xcal = X.iloc[:cut], X.iloc[cut:]
        else:
            Xtr, Xcal = X[:cut], X[cut:]
        ytr, ycal = y[:cut], y[cut:]

        # If the calibration slice is too small or single-class, skip calibration
        if (len(ycal) < self.min_calib or len(np.unique(ycal)) < 2
                or ycal.sum() < self.min_calib_pos):
            self.model_ = self.base_factory().fit(X, y)
            self.calibrated_ = False
            return self

        base = self.base_factory().fit(Xtr, ytr)

        # isotonic needs a fair number of positives; fall back to sigmoid if sparse
        method = self.method
        if method == "isotonic" and ycal.sum() < 25:
            method = "sigmoid"

        try:
            cal = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
            cal.fit(Xcal, ycal)
            self.model_ = cal
            self.calibrated_ = True
        except Exception:
            self.model_ = self.base_factory().fit(X, y)
            self.calibrated_ = False
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

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
