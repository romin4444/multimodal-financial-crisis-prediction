"""
v3 — Causal (filtered) HMM regime probabilities.

PROBLEM IN v2:
    `model.predict_proba(X)` returns the SMOOTHED posterior P(state_t | x_1..x_T),
    computed by the forward-BACKWARD algorithm. The backward pass uses the ENTIRE
    sequence — including data AFTER time t. So the v2 "regime at time t" silently
    incorporates the future. The headline "detected GFC 37 days early" is partly an
    artifact: the model labels July 2008 as Crisis using knowledge of the September
    crash. That is not a real-time signal.

FIX:
    Compute the FILTERED posterior P(state_t | x_1..x_t) using only the forward
    pass. At every t this uses strictly past-and-present data, so it is a genuine
    online / real-time regime estimate suitable for early-warning deployment.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def filtered_state_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Forward-only (causal) state posteriors for a fitted GaussianHMM.

    Returns array of shape (n_samples, n_states) where row t is
    P(state_t | x_1, ..., x_t) — no future information used.
    """
    n = X.shape[0]
    k = model.n_components

    # Emission log-likelihoods per state (Gaussian, full covariance)
    log_b = np.empty((n, k))
    for s in range(k):
        log_b[:, s] = multivariate_normal.logpdf(
            X, mean=model.means_[s], cov=model.covars_[s], allow_singular=True
        )

    log_pi = np.log(model.startprob_ + 1e-300)
    log_A = np.log(model.transmat_ + 1e-300)

    log_alpha = np.empty((n, k))
    log_alpha[0] = log_pi + log_b[0]
    log_alpha[0] -= logsumexp(log_alpha[0])  # normalise → filtered posterior at t=0

    for t in range(1, n):
        # predict step: marginalise previous state
        pred = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)  # (k,)
        log_alpha[t] = pred + log_b[t]
        log_alpha[t] -= logsumexp(log_alpha[t])  # normalise → filtered posterior

    return np.exp(log_alpha)


def volatility_state_order(model, feature_cols: List[str]) -> List[int]:
    """
    Rank raw HMM states by mean volatility so that 0=Stable, 1=Volatile, 2=Crisis.
    Mirrors the v2 labelling convention for comparability.
    """
    d = min(len(feature_cols), model.means_.shape[1])
    means = pd.DataFrame(model.means_[:, :d], columns=feature_cols[:d])
    vol_col = "vol_21d" if "vol_21d" in means.columns else means.columns[0]
    return list(means[vol_col].argsort().values)


def causal_regime_frame(
    model,
    X: np.ndarray,
    index: pd.DatetimeIndex,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Build a DataFrame of causal regime probabilities + hard label, with states
    ordered by volatility (Stable/Volatile/Crisis).
    """
    filt = filtered_state_proba(model, X)
    order = volatility_state_order(model, feature_cols)
    state_map = {int(order[i]): i for i in range(len(order))}

    k = filt.shape[1]
    ordered = np.zeros_like(filt)
    for raw_state, ranked in state_map.items():
        if ranked < k:
            ordered[:, ranked] = filt[:, raw_state]

    name_map = {0: "c_prob_stable", 1: "c_prob_volatile", 2: "c_prob_crisis"}
    data = {name_map.get(i, f"c_prob_s{i}"): ordered[:, i] for i in range(k)}
    data["c_regime"] = ordered.argmax(axis=1)
    return pd.DataFrame(data, index=index)
