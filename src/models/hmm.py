"""
Hidden Markov Model regime detection.
n=3 canonical (stable/volatile/crisis) following Ang & Timmermann (2012).
BIC profile across n∈{2,3,4} reported for transparency.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("models.hmm")

# State label → human-readable name
STATE_NAMES = {0: "Stable", 1: "Volatile", 2: "Crisis"}


class HMMResult:
    """Fitted HMM with labelled states."""

    def __init__(
        self,
        model: GaussianHMM,
        scaler: StandardScaler,
        feature_cols: List[str],
        log_likelihood: float,
        bic: float,
        n_states: int,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.log_likelihood = log_likelihood
        self.bic = bic
        self.n_states = n_states

    def predict_regime(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with 'regime' and posterior probability columns."""
        fcols = [c for c in self.feature_cols if c in feat.columns]
        Xdf = feat[fcols].dropna()
        X = self.scaler.transform(Xdf)
        labels, probs, _ = _label_states(self.model, X, fcols)

        k = self.model.n_components
        d: dict = {"regime": labels}
        name_map = {0: "prob_stable", 1: "prob_volatile", 2: "prob_crisis"}
        for i in range(k):
            col = name_map.get(i, f"prob_s{i}")
            d[col] = probs[:, i] if i < probs.shape[1] else 0.0

        rdf = pd.DataFrame(d, index=Xdf.index)
        for s, nm in STATE_NAMES.items():
            pct = (rdf["regime"] == s).mean() * 100
            log.info("Regime distribution", extra={"state": nm, "pct": round(pct, 2)})
        return rdf

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        log.info("HMM saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: Path) -> "HMMResult":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected HMMResult, got {type(obj)}")
        return obj


def select_and_fit_hmm(feat: pd.DataFrame) -> Tuple[HMMResult, Dict[int, HMMResult]]:
    """
    Fit HMMs for n_states in cfg.hmm.n_states_candidates.
    Retain canonical n=3 model regardless of BIC minimum.

    Returns:
        (canonical_result, all_results_by_n)
    """
    log.info(
        "HMM regime selection",
        extra={"candidates": cfg.hmm.n_states_candidates, "canonical": cfg.hmm.canonical_n_states},
    )
    fcols = [c for c in cfg.hmm.features if c in feat.columns]
    if len(fcols) < 2:
        raise ValueError(f"Too few HMM feature columns available: {fcols}")

    Xdf = feat[fcols].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(Xdf)

    all_results: Dict[int, HMMResult] = {}
    for n in cfg.hmm.n_states_candidates:
        try:
            result = _fit_best_hmm(X, n, scaler, fcols)
            all_results[n] = result
        except RuntimeError as exc:
            log.warning("HMM failed for n_states", extra={"n": n, "error": str(exc)})

    if not all_results:
        raise RuntimeError("All HMM configurations failed to converge")

    bic_min_n = min(all_results, key=lambda k: all_results[k].bic)
    log.info("BIC profile", extra={n: round(r.bic, 2) for n, r in all_results.items()})
    log.info("BIC-optimal n", extra={"n": bic_min_n})

    canonical_n = cfg.hmm.canonical_n_states
    if canonical_n not in all_results:
        log.warning(
            "Canonical n not in results — using BIC-optimal",
            extra={"canonical_n": canonical_n, "fallback_n": bic_min_n},
        )
        canonical_n = bic_min_n

    canonical = all_results[canonical_n]
    canonical.save(cfg.paths.model_dir / "hmm_best.pkl")
    return canonical, all_results


# ─── Private ─────────────────────────────────────────────────────────────────

def _fit_best_hmm(
    X: np.ndarray,
    n: int,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> HMMResult:
    """Try cfg.hmm.n_init_seeds random seeds — return best by log-likelihood."""
    best_model: Optional[GaussianHMM] = None
    best_ll: float = -np.inf

    for seed in range(cfg.hmm.n_init_seeds):
        try:
            m = GaussianHMM(
                n_components=n,
                covariance_type="full",
                n_iter=cfg.hmm.n_iter,
                tol=cfg.hmm.tol,
                random_state=seed,
            )
            m.fit(X)
            ll = m.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, m
        except Exception:
            pass

    if best_model is None:
        raise RuntimeError(f"All {cfg.hmm.n_init_seeds} seeds failed for n={n}")

    bic = _compute_bic(best_model, X)
    log.info("HMM fitted", extra={"n": n, "ll": round(best_ll, 2), "bic": round(bic, 2)})
    return HMMResult(
        model=best_model,
        scaler=scaler,
        feature_cols=feature_cols,
        log_likelihood=best_ll,
        bic=bic,
        n_states=n,
    )


def _compute_bic(model: GaussianHMM, X: np.ndarray) -> float:
    """Correct BIC for full-covariance GaussianHMM."""
    n, d = X.shape
    k = model.n_components
    n_params = k * (k - 1) + k * d + k * d * (d + 1) // 2 + (k - 1)
    return -2 * model.score(X) + n_params * np.log(n)


def _label_states(
    model: GaussianHMM, X: np.ndarray, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Rank states by mean volatility (vol_21d if present, else first feature).
    0 = Stable | 1 = Volatile | 2 = Crisis
    """
    k = model.n_components
    d = min(len(feature_cols), X.shape[1])
    means = pd.DataFrame(model.means_[:, :d], columns=feature_cols[:d])
    vol_col = "vol_21d" if "vol_21d" in means.columns else means.columns[0]
    order = means[vol_col].argsort().values
    state_map = {int(order[i]): i for i in range(k)}

    raw_labels = model.predict(X)
    labels = np.vectorize(state_map.get)(raw_labels)

    raw_probs = model.predict_proba(X)
    probs = np.zeros_like(raw_probs)
    for rs, ss in state_map.items():
        if ss < probs.shape[1]:
            probs[:, ss] = raw_probs[:, rs]

    return labels, probs, state_map
