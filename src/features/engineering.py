"""
Price-based feature engineering.
All computed features are deterministic functions of raw OHLCV data.
Includes ADF stationarity and ARCH-LM heteroskedasticity diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller

from src.logging_setup import get_logger

log = get_logger("features.engineering")


def engineer_features(sp500: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all price-based features from S&P 500 and VIX data.

    Returns a DataFrame indexed on trading days with columns:
        close, volume, vix, log_ret, vol_5d/21d/63d/126d, drawdown_63,
        vix_chg, vix_ma21, vix_spike, mom_5d/21d/63d, vol_ratio, garch_var
    """
    log.info("Engineering features")
    df = pd.DataFrame(index=sp500.index)
    df["close"] = sp500["Close"]
    df["volume"] = sp500["Volume"]
    df["vix"] = vix["Close"].reindex(df.index).ffill()

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    for w in [5, 21, 63, 126]:
        df[f"vol_{w}d"] = df["log_ret"].rolling(w).std() * np.sqrt(252)

    df["drawdown_63"] = (
        df["close"]
        .rolling(63)
        .apply(_max_drawdown, raw=True)
    )

    df["vix_chg"] = df["vix"].pct_change()
    df["vix_ma21"] = df["vix"].rolling(21).mean()
    df["vix_spike"] = (
        df["vix"] > df["vix"].rolling(63).mean() + 2 * df["vix"].rolling(63).std()
    ).astype(int)

    for d in [5, 21, 63]:
        df[f"mom_{d}d"] = df["close"].pct_change(d)

    df["vol_ratio"] = df["volume"] / df["volume"].rolling(21).mean()
    df["garch_var"] = np.nan  # placeholder; filled by models.garch

    df = df.dropna(subset=["log_ret"])
    _run_diagnostics(df["log_ret"].dropna())

    log.info("Feature matrix ready", extra={"shape": list(df.shape), "columns": list(df.columns)})
    return df


def validate_features(df: pd.DataFrame) -> None:
    """
    Assert invariants that must hold on the feature matrix.
    Call before fitting models to catch data corruption early.
    """
    required = {"close", "log_ret", "vol_21d", "vix", "drawdown_63"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature matrix missing required columns: {missing}")

    if df["log_ret"].isna().mean() > 0.05:
        raise ValueError("log_ret has >5%% missing values — check raw data")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Feature index is not sorted ascending")

    if df.index.duplicated().any():
        raise ValueError("Feature index contains duplicate dates")


# ─── Private ─────────────────────────────────────────────────────────────────

def _max_drawdown(x: np.ndarray) -> float:
    peak = x.max()
    if peak == 0:
        return 0.0
    return (x[-1] - peak) / peak


def _run_diagnostics(returns: pd.Series) -> None:
    adf_stat, adf_p, *_ = adfuller(returns, autolag="AIC")
    arch_stat, arch_p, *_ = het_arch(returns)
    log.info(
        "Return diagnostics",
        extra={
            "adf_stat": round(float(adf_stat), 4),
            "adf_p": round(float(adf_p), 6),
            "adf_stationary": bool(adf_p < 0.05),
            "arch_stat": round(float(arch_stat), 4),
            "arch_p": round(float(arch_p), 6),
            "arch_effects_present": bool(arch_p < 0.05),
        },
    )
    if adf_p >= 0.05:
        log.warning("Returns may be non-stationary (ADF p >= 0.05) — GARCH assumptions may be violated")
    if arch_p >= 0.05:
        log.warning("No significant ARCH effects detected — GARCH modeling may not be necessary")
