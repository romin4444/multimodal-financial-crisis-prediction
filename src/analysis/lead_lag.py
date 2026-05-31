"""
Lead-lag cross-correlation and Granger causality analysis.
Sentiment → price-regime causal direction (Bollen et al. 2011 framework).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("analysis.lead_lag")


def cross_corr(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = None,
    n_boot: int = None,
) -> dict:
    """
    Pearson cross-correlation at lags -max_lag to +max_lag.
    Positive lag k: y leads x by k days.
    Bootstrap 95% CI on the peak-lag estimate.
    """
    max_lag = max_lag or cfg.lead_lag.max_lag_days
    n_boot = n_boot or cfg.lead_lag.bootstrap_n

    idx = x.index.intersection(y.index)
    xv = x.reindex(idx).ffill().fillna(0).values.astype(float)
    yv = y.reindex(idx).ffill().fillna(0).values.astype(float)
    n = len(idx)
    lags = np.arange(-max_lag, max_lag + 1)

    corrs = np.array([_pearson_at_lag(xv, yv, n, lag) for lag in lags])

    peak_idx = int(np.argmax(np.abs(corrs)))
    peak_lag = int(lags[peak_idx])
    peak_r = float(corrs[peak_idx])

    # Block bootstrap (Politis & Romano 1994) — preserves time-series structure.
    # Naive iid resampling destroys autocorrelation and collapses peak lag to 0.
    rng = np.random.default_rng(cfg.seed)
    boot_lags = []
    block_size = max(20, int(n ** (1 / 3)))  # standard rule of thumb
    n_blocks = (n // block_size) + 1
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size, size=n_blocks)
        idx_ = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        xb, yb = xv[idx_], yv[idx_]
        bc = np.array([_pearson_at_lag(xb, yb, n, lag) for lag in lags])
        bc = np.where(np.isfinite(bc), bc, 0.0)
        boot_lags.append(int(lags[np.argmax(np.abs(bc))]))

    ci_lo = float(np.percentile(boot_lags, 2.5))
    ci_hi = float(np.percentile(boot_lags, 97.5))

    interp = _interpret_lag(peak_lag)
    return dict(
        lags=lags, corrs=corrs, peak_lag=peak_lag,
        peak_r=peak_r, ci_lo=ci_lo, ci_hi=ci_hi, interp=interp,
    )


def run_all_lead_lag(feat: pd.DataFrame, sent: pd.DataFrame) -> Dict[str, dict]:
    """Compute cross-correlation for full sample and each crisis window."""
    fsi = feat["FSI"]
    fear = sent["fear_index"]
    results = {"overall": cross_corr(fsi, fear)}
    log.info("Lead-lag overall", extra={"interp": results["overall"]["interp"], "peak_r": round(results["overall"]["peak_r"], 4)})

    for name, (cs, ce) in cfg.crisis_window_tuples.items():
        pre = pd.Timestamp(cs) - pd.DateOffset(months=6)
        fsi_w = fsi[(fsi.index >= pre) & (fsi.index <= ce)]
        fear_w = fear[(fear.index >= pre) & (fear.index <= ce)]
        if len(fsi_w) < 60 or len(fear_w) < 20:
            continue
        r = cross_corr(fsi_w, fear_w, n_boot=cfg.lead_lag.crisis_window_bootstrap_n)
        results[name] = r
        log.info("Lead-lag crisis window", extra={"crisis": name, "interp": r["interp"], "peak_r": round(r["peak_r"], 4)})

    return results


def run_granger(fsi: pd.Series, fear: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """
    Test whether fear Granger-causes FSI at lags 1..max_lag.
    Replicates Bollen et al. (2011).
    """
    idx = fsi.index.intersection(fear.index)
    data = pd.DataFrame({"fsi": fsi.loc[idx], "fear": fear.loc[idx]}).dropna()
    rows = []
    try:
        gc = grangercausalitytests(data[["fsi", "fear"]], maxlag=max_lag, verbose=False)
        for lag, res in gc.items():
            f_stat, p_val = res[0]["params_ftest"][:2]
            rows.append({
                "lag": lag,
                "f_stat": round(float(f_stat), 4),
                "p_value": round(float(p_val), 4),
                "significant": p_val < 0.05,
            })
    except Exception as exc:
        log.warning("Granger causality test failed", extra={"error": str(exc)})

    df = pd.DataFrame(rows)
    sig_lags = df[df["significant"]]["lag"].tolist() if not df.empty else []
    log.info("Granger causality", extra={"significant_lags": sig_lags})
    return df


# ─── Private ─────────────────────────────────────────────────────────────────

def _pearson_at_lag(xv: np.ndarray, yv: np.ndarray, n: int, lag: int) -> float:
    if lag >= 0 and n > lag:
        r = np.corrcoef(xv[lag:], yv[: n - lag])[0, 1]
    elif lag < 0 and n > -lag:
        r = np.corrcoef(xv[: n + lag], yv[-lag:])[0, 1]
    else:
        return 0.0
    return float(r) if np.isfinite(r) else 0.0


def _interpret_lag(peak_lag: int) -> str:
    if peak_lag > 0:
        return f"Sentiment LEADS price-regime by {peak_lag} trading days"
    if peak_lag < 0:
        return f"Price-regime LEADS sentiment by {abs(peak_lag)} trading days"
    return "Contemporaneous (peak lag = 0)"
