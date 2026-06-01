"""
v3 — Probabilistic & Deflated Sharpe Ratio (Bailey & López de Prado).

WHY (the 2025/2026 bar):
    When you try many models/strategies and report the best, the headline Sharpe
    is selection-biased upward. The Deflated Sharpe Ratio (DSR) corrects for:
      (1) the NUMBER of trials, (2) non-normal returns (skew/kurtosis), and
      (3) sample length. It is the standard a modern reviewer expects before
    believing any backtest. We report it so our "best model" claims are honest.

References:
    Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

_EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr: float, n: int, sr_benchmark: float = 0.0,
    gamma3: float = 0.0, gamma4: float = 3.0,
) -> float:
    """
    PSR = P(true SR > benchmark) given observed (non-annualised) Sharpe `sr`,
    sample length `n`, and return skew/kurtosis.
    """
    denom = np.sqrt(max(1e-12, 1 - gamma3 * sr + (gamma4 - 1) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * np.sqrt(max(1, n - 1)) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """
    Expected maximum Sharpe under the null (no skill), given the cross-sectional
    variance of the per-trial Sharpe estimates and the number of trials N.
    This is the DSR benchmark SR0.
    """
    t = np.asarray(trial_sharpes, dtype=float)
    t = t[np.isfinite(t)]
    n = len(t)
    if n < 2:
        return 0.0
    var_sr = float(np.var(t, ddof=1))
    if var_sr <= 0:
        return 0.0
    sigma = np.sqrt(var_sr)
    inv = norm.ppf
    sr0 = sigma * (
        (1 - _EULER_MASCHERONI) * inv(1 - 1.0 / n)
        + _EULER_MASCHERONI * inv(1 - 1.0 / (n * np.e))
    )
    return float(sr0)


def deflated_sharpe_ratio(
    selected_returns: np.ndarray,
    trial_sharpes: np.ndarray,
) -> dict:
    """
    Deflated Sharpe Ratio for the SELECTED strategy among `trial_sharpes` trials.

    Args:
        selected_returns: per-period returns of the chosen (best) strategy.
        trial_sharpes: per-period Sharpe of EVERY strategy tried (incl. selected).

    Returns dict with observed SR, benchmark SR0, and DSR = P(SR is real).
    """
    r = np.asarray(selected_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return {"n": int(n), "sharpe": None, "sr_benchmark": None, "dsr": None}

    mu, sd = r.mean(), r.std(ddof=1)
    sr = float(mu / sd) if sd > 0 else 0.0
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # non-excess (normal = 3)
    sr0 = expected_max_sharpe(trial_sharpes)
    dsr = probabilistic_sharpe_ratio(sr, n, sr_benchmark=sr0, gamma3=g3, gamma4=g4)

    return {
        "n": int(n),
        "n_trials": int(np.sum(np.isfinite(np.asarray(trial_sharpes, dtype=float)))),
        "sharpe_per_period": round(sr, 4),
        "sharpe_annualised": round(sr * np.sqrt(252), 4),
        "sr_benchmark": round(sr0, 4),
        "skew": round(g3, 4),
        "kurtosis": round(g4, 4),
        "deflated_sharpe": round(dsr, 4),
        "is_significant_5pct": bool(dsr > 0.95),
    }
