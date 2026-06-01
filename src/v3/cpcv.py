"""
v3 — Combinatorial Purged Cross-Validation (CPCV) + Probability of Backtest
Overfitting (PBO / CSCV).  López de Prado, "Advances in Financial ML" (2018).

WHY (the 2025/2026 bar):
    Recent work (Backtest Overfitting in the ML Era, 2024) shows CPCV is markedly
    superior to plain walk-forward at NOT being fooled by overfit configurations,
    as measured by a lower Probability of Backtest Overfitting. This module brings
    the project's evaluation up to that standard:

      - CPCV: split the timeline into N contiguous groups, use every C(N,k)
        combination of k groups as the test set, PURGE training samples whose
        forward-looking label window overlaps a test group, and apply an EMBARGO.
        Each sample is predicted on multiple paths -> many backtest paths.
      - PBO (CSCV): given a (time-slice x strategy) performance matrix, estimate
        the probability that the in-sample-best strategy underperforms the median
        out-of-sample -> the chance your "winner" is an overfit artefact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass
class CPCVConfig:
    n_groups: int = 6          # N contiguous time groups
    n_test_groups: int = 2     # k groups held out per combination -> C(N,k) splits
    embargo: int = 21          # trading days purged around each test block (= label horizon)


@dataclass
class CPCVResult:
    oos_proba: pd.Series                  # mean OOS probability per sample (across paths)
    oos_label: pd.Series
    n_splits: int
    per_split_proba: List[pd.Series] = field(default_factory=list)


def cpcv_splits(n: int, cfg: CPCVConfig):
    """Yield (train_idx, test_idx) positional arrays for each C(N,k) combination,
    with purge + embargo applied to the training set."""
    groups = np.array_split(np.arange(n), cfg.n_groups)
    for combo in combinations(range(cfg.n_groups), cfg.n_test_groups):
        test_idx = np.concatenate([groups[g] for g in combo])
        test_lo, test_hi = test_idx.min(), test_idx.max()
        # Train = everything not in a test group, minus an embargo around the test span
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        lo = max(0, test_lo - cfg.embargo)
        hi = min(n - 1, test_hi + cfg.embargo)
        train_mask[lo : hi + 1] = False  # purge the embargo band around the test block
        train_idx = np.where(train_mask)[0]
        yield train_idx, test_idx


def run_cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], object],
    cfg: CPCVConfig = CPCVConfig(),
) -> CPCVResult:
    """
    Run CPCV. Returns mean out-of-sample probability per sample (averaged over the
    multiple paths each sample appears in) plus the per-split probability series.
    """
    n = len(X)
    dates = X.index
    proba_sum = pd.Series(0.0, index=dates)
    proba_cnt = pd.Series(0.0, index=dates)
    per_split: List[pd.Series] = []
    n_splits = 0

    for train_idx, test_idx in cpcv_splits(n, cfg):
        ytr = y.iloc[train_idx]
        valid = ytr.notna().to_numpy()
        if valid.sum() < 50 or ytr[valid].sum() < 3:
            continue
        Xtr = X.iloc[train_idx][valid]
        ytr_v = ytr[valid].astype(int)
        try:
            model = model_factory().fit(Xtr, ytr_v)
            p = model.predict_proba(X.iloc[test_idx])[:, 1]
        except Exception:
            continue
        s = pd.Series(p, index=dates[test_idx])
        per_split.append(s)
        proba_sum.iloc[test_idx] += p
        proba_cnt.iloc[test_idx] += 1
        n_splits += 1

    oos = (proba_sum / proba_cnt.replace(0, np.nan)).rename("cpcv_oos_proba")
    return CPCVResult(oos_proba=oos, oos_label=y, n_splits=n_splits, per_split_proba=per_split)


def probability_of_backtest_overfitting(perf_matrix: np.ndarray, n_partitions: int = 10) -> dict:
    """
    CSCV estimate of PBO (Bailey, Borwein, López de Prado, Zhu 2017).

    Args:
        perf_matrix: shape (T_slices, N_strategies) of a performance statistic
            (e.g. average precision per time-slice per strategy). Higher = better.
        n_partitions: S, number of equal time-slices (must be even). C(S, S/2)
            in-sample/out-of-sample partitions are evaluated.

    Returns dict with pbo (probability the IS-best is below OOS median) and the
    mean OOS rank of the IS-best strategy.
    """
    M = np.asarray(perf_matrix, dtype=float)
    M = M[~np.isnan(M).any(axis=1)]  # drop slices with any NaN
    T, N = M.shape
    if T < 4 or N < 2:
        return {"pbo": float("nan"), "n_strategies": int(N), "note": "insufficient data"}

    S = min(n_partitions, T - (T % 2))
    if S % 2 == 1:
        S -= 1
    if S < 2:
        return {"pbo": float("nan"), "n_strategies": int(N), "note": "too few slices"}

    slices = np.array_split(np.arange(T), S)
    logits = []
    for is_combo in combinations(range(S), S // 2):
        is_rows = np.concatenate([slices[i] for i in is_combo])
        oos_rows = np.concatenate([slices[i] for i in range(S) if i not in is_combo])
        is_perf = M[is_rows].mean(axis=0)
        oos_perf = M[oos_rows].mean(axis=0)
        best_is = int(np.argmax(is_perf))
        # relative rank of the IS-best strategy out-of-sample (1 = worst .. N = best)
        oos_ranks = rankdata(oos_perf)
        rel = oos_ranks[best_is] / (N + 1)
        rel = min(max(rel, 1e-6), 1 - 1e-6)
        logits.append(np.log(rel / (1 - rel)))

    logits = np.asarray(logits)
    pbo = float(np.mean(logits <= 0.0))  # P(IS-best lands below OOS median)
    return {
        "pbo": round(pbo, 4),
        "n_strategies": int(N),
        "n_partitions": int(S),
        "n_combinations": int(len(logits)),
        "mean_logit": round(float(logits.mean()), 4),
    }


def per_slice_performance(
    oos_series: dict, y: pd.Series, n_slices: int = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a (n_slices x n_strategies) performance matrix (average precision per
    contiguous time slice) for PBO from a dict of {name: oos_proba_series}.
    """
    from sklearn.metrics import average_precision_score

    names = list(oos_series.keys())
    common = None
    for s in oos_series.values():
        idx = s.dropna().index
        common = idx if common is None else common.intersection(idx)
    common = common.intersection(y.dropna().index)
    if common is None or len(common) < n_slices * 5:
        return np.empty((0, len(names))), names

    common = common.sort_values()
    slices = np.array_split(np.arange(len(common)), n_slices)
    mat = np.full((n_slices, len(names)), np.nan)
    yv = y.reindex(common)
    for j, name in enumerate(names):
        pv = oos_series[name].reindex(common)
        for i, sl in enumerate(slices):
            ys = yv.iloc[sl].to_numpy()
            ps = pv.iloc[sl].to_numpy()
            m = ~(np.isnan(ys) | np.isnan(ps))
            if m.sum() >= 5 and len(np.unique(ys[m])) > 1:
                mat[i, j] = average_precision_score(ys[m], ps[m])
    return mat, names
