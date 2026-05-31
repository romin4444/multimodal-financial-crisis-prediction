"""
v3 — "Walk-forward everything": online (per-fold refit) regime features.

PROBLEM:
    Even in v3, the HMM and the FSI scaler were fit ONCE on a train-head and then
    applied to the whole series. That is still a mild in-sample contamination of
    the feature generators (parameters chosen with knowledge of the full period's
    statistics), so the backtest is "mostly honest" rather than fully deployable.

FIX:
    Refit the StandardScaler + HMM on an EXPANDING window at regular boundaries.
    For each block [b, b+step) the regime probabilities are produced by a model
    trained only on data BEFORE b, and computed with the forward-only filter
    (src/v3/causal_regime.filtered_state_proba) so each day uses only its past.
    Result: the regime feature at every test day is fully out-of-sample.

This is the deployable version: at time b you would have exactly this model.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.logging_setup import get_logger
from src.v3.causal_regime import filtered_state_proba, volatility_state_order

log = get_logger("v3.online")


def _fit_hmm(X: np.ndarray, n_states: int, n_seeds: int, n_iter: int):
    best, best_ll = None, -np.inf
    for seed in range(n_seeds):
        try:
            m = GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=n_iter, random_state=seed)
            m.fit(X)
            ll = m.score(X)
            if ll > best_ll:
                best_ll, best = ll, m
        except Exception:
            pass
    return best


def compute_online_regime(
    feat: pd.DataFrame,
    hmm_feature_cols: List[str],
    min_train: int = 1260,
    refit_step: int = 252,
    n_states: int = 3,
    n_seeds: int = 6,
    n_iter: int = 80,
) -> pd.DataFrame:
    """
    Produce causal, out-of-sample regime posteriors by refitting the scaler+HMM
    on an expanding window every `refit_step` rows.

    Returns DataFrame indexed like feat[hmm_feature_cols].dropna() with columns
    o_prob_stable / o_prob_volatile / o_prob_crisis and o_regime.
    """
    cols = [c for c in hmm_feature_cols if c in feat.columns]
    X = feat[cols].dropna()
    idx = X.index
    Xv = X.to_numpy(dtype=float)
    n = len(Xv)

    out = np.full((n, n_states), np.nan)
    n_refits = 0

    b = max(min_train, refit_step)
    # Warmup region [0:b] filled later with the first fitted model (training-only rows)
    first_model = None
    first_scaler = None

    while b < n:
        end = min(b + refit_step, n)
        scaler = StandardScaler().fit(Xv[:b])
        model = _fit_hmm(scaler.transform(Xv[:b]), n_states, n_seeds, n_iter)
        if model is None:
            b = end
            continue
        n_refits += 1
        if first_model is None:
            first_model, first_scaler = model, scaler

        # Causal filtered posteriors over [:end]; keep only the OOS block [b:end]
        filt = filtered_state_proba(model, scaler.transform(Xv[:end]))
        order = volatility_state_order(model, cols)
        state_map = {int(order[i]): i for i in range(n_states)}
        ordered = np.zeros_like(filt)
        for raw, ranked in state_map.items():
            if ranked < ordered.shape[1]:
                ordered[:, ranked] = filt[:, raw]
        out[b:end] = ordered[b:end]
        b = end

    # Fill warmup [0:min_train_boundary] with the first model (used only in training folds)
    if first_model is not None:
        warm_end = np.where(~np.isnan(out[:, 0]))[0]
        warm_end = warm_end[0] if len(warm_end) else n
        filt0 = filtered_state_proba(first_model, first_scaler.transform(Xv[:warm_end]))
        order0 = volatility_state_order(first_model, cols)
        smap0 = {int(order0[i]): i for i in range(n_states)}
        for raw, ranked in smap0.items():
            if ranked < out.shape[1]:
                out[:warm_end, ranked] = filt0[:warm_end, raw]

    names = {0: "o_prob_stable", 1: "o_prob_volatile", 2: "o_prob_crisis"}
    data = {names.get(i, f"o_prob_s{i}"): out[:, i] for i in range(n_states)}
    df = pd.DataFrame(data, index=idx)
    df["o_regime"] = np.nan_to_num(out, nan=0.0).argmax(axis=1)
    log.info("Online regime computed", extra={"refits": n_refits, "rows": n})
    return df
