"""
v3 — Purged, embargoed walk-forward backtesting.

PROBLEM IN v2:
    "Event-based holdout" trained on all non-crisis days and tested on the three
    known crisis windows. Two fatal flaws:
      1. You cannot know in advance which periods are crises, so this is not a
         deployable evaluation — it peeks at the answer to design the split.
      2. The fixed scalers / FSI were fit on the full sample (incl. test windows).

FIX:
    Expanding-window walk-forward (the only honest way to evaluate a time-series
    early-warning system):
      - At each step, train on [0, train_end), predict the next block.
      - EMBARGO of `horizon` days between train and test (purging, López de Prado
        2018) so the forward-looking label window cannot overlap training.
      - All preprocessing (scaling) is fit inside the fold on training rows only.
      - Out-of-sample predictions are stitched into one series and scored ONCE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    min_train: int = 1260          # ~5 years before first prediction
    step: int = 63                 # refit cadence (~quarterly)
    embargo: int = 21              # = label horizon; purge overlap
    horizon: int = 21             # forward label horizon (for bookkeeping)


@dataclass
class WalkForwardResult:
    oos_proba: pd.Series
    oos_label: pd.Series
    n_folds: int
    fold_log: List[dict] = field(default_factory=list)


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], object],
    config: WalkForwardConfig,
) -> WalkForwardResult:
    """
    Run expanding-window walk-forward and return stitched out-of-sample
    probabilities aligned to `X.index`.

    `model_factory()` must return a fresh estimator each call with
    .fit(X, y) and .predict_proba(X) -> (n, 2).
    """
    n = len(X)
    dates = X.index
    oos = pd.Series(np.nan, index=dates, name="oos_proba")
    fold_log: List[dict] = []

    train_end = config.min_train
    n_folds = 0

    while train_end + config.embargo < n:
        test_start = train_end + config.embargo
        test_end = min(test_start + config.step, n)

        tr_idx = slice(0, train_end)
        te_idx = slice(test_start, test_end)

        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xte = X.iloc[te_idx]

        # Drop rows with NaN label in training (right-edge / warmup)
        valid = ytr.notna().to_numpy()
        Xtr_v, ytr_v = Xtr[valid], ytr[valid]

        if ytr_v.notna().sum() < 50 or ytr_v.sum() < 3:
            # not enough signal yet — skip, advance window
            train_end += config.step
            continue

        model = model_factory()
        try:
            model.fit(Xtr_v, ytr_v.astype(int))
            proba = model.predict_proba(Xte)[:, 1]
            oos.iloc[te_idx] = proba
            n_folds += 1
            fold_log.append({
                "train_end": str(dates[train_end - 1].date()),
                "test_start": str(dates[test_start].date()),
                "test_end": str(dates[test_end - 1].date()),
                "train_pos_rate": round(float(ytr_v.mean()), 4),
            })
        except Exception as exc:  # noqa: BLE001
            fold_log.append({"train_end": str(dates[train_end - 1].date()), "error": str(exc)})

        train_end += config.step

    return WalkForwardResult(
        oos_proba=oos,
        oos_label=y,
        n_folds=n_folds,
        fold_log=fold_log,
    )
