"""
v3 — VIX-orthogonal macro/stress features.

PROBLEM:
    The model loses to a simple VIX threshold because most of its signal *is*
    VIX in disguise (FSI is 30% VIX; synthetic sentiment is a VIX transform).
    The only way to beat VIX is to add information NOT already in VIX.

FIX:
    Build features from sources orthogonal to option-implied equity vol:
      - Credit: high-yield OAS level + momentum (FRED BAMLH0A0HYM2).
      - Rates : yield-curve slope (T10Y2Y) level + change; term structure.
      - Policy: fed-funds change (monetary tightening pace).
      - Cross-asset: oil momentum; average pairwise equity correlation
        (correlation spikes = diversification breakdown = stress).
    All features are causal (differences / rolling stats use only past data).
    FRED series are forward-filled with a cap to avoid stale-value propagation.

NOTE on options skew: genuinely VIX-orthogonal but needs an options dataset we
don't have here; left as documented future work in PROJECT_REVIEW_AND_ROADMAP.md.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("v3.macro")

MACRO_FEATURE_COLS = [
    "credit_spread", "credit_chg_63",
    "yield_slope", "yield_slope_chg_63",
    "fedfunds_chg_126", "oil_mom_63", "xasset_corr",
]


def cross_asset_correlation(market: Dict[str, pd.DataFrame], window: int = 63) -> pd.Series:
    """
    Average pairwise rolling correlation of daily returns across the individual
    stocks. Spikes toward 1.0 signal stress (everything moves together). This is
    information NOT contained in the VIX level.
    """
    rets = {}
    for key, df in market.items():
        if key in ("sp500", "vix"):
            continue
        if "Close" in df.columns:
            rets[key] = np.log(df["Close"] / df["Close"].shift(1))
    if len(rets) < 2:
        return pd.Series(dtype=float, name="xasset_corr")

    R = pd.DataFrame(rets).dropna(how="all")
    cols = list(R.columns)
    pair_corrs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair_corrs.append(R[cols[i]].rolling(window).corr(R[cols[j]]))
    avg = pd.concat(pair_corrs, axis=1).mean(axis=1)
    return avg.rename("xasset_corr")


def build_macro_features(
    fred_daily: pd.DataFrame,
    market: Dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Assemble VIX-orthogonal macro features aligned to the trading-day `index`.
    Robust to missing FRED columns (only builds what's available).
    """
    df = pd.DataFrame(index=index)
    limit = cfg.data.fred_ffill_limit

    def _ff(col: str) -> pd.Series:
        return fred_daily[col].reindex(index).ffill(limit=limit) if col in fred_daily.columns \
            else pd.Series(np.nan, index=index)

    # Credit (high-yield OAS)
    cs = _ff("credit_spread")
    df["credit_spread"] = cs
    df["credit_chg_63"] = cs.diff(63)

    # Yield-curve slope (T10Y2Y)
    ys = _ff("yield_spread")
    df["yield_slope"] = ys
    df["yield_slope_chg_63"] = ys.diff(63)

    # Term structure fallback from raw treasuries if slope missing
    if df["yield_slope"].notna().sum() == 0:
        t10, t2 = _ff("treasury_10y"), _ff("treasury_2y")
        df["yield_slope"] = t10 - t2
        df["yield_slope_chg_63"] = (t10 - t2).diff(63)

    # Monetary policy pace
    ff = _ff("fed_funds")
    df["fedfunds_chg_126"] = ff.diff(126)

    # Cross-asset: oil momentum
    oil = _ff("oil_price")
    df["oil_mom_63"] = oil.pct_change(63)

    # Cross-asset correlation spike
    xc = cross_asset_correlation(market)
    df["xasset_corr"] = xc.reindex(index).ffill()

    avail = [c for c in MACRO_FEATURE_COLS if c in df.columns and df[c].notna().sum() > 100]
    log.info("Macro (VIX-orthogonal) features built", extra={"available": avail})
    return df


def usable_macro_cols(feat: pd.DataFrame, min_nonnull_frac: float = 0.5) -> List[str]:
    """
    Return the macro feature columns with adequate historical coverage.

    The bundled fred_data.csv cache has `credit_spread` (HY OAS) populated only
    for recent dates, so it is auto-dropped here. Set FRED_API_KEY and delete
    data/cache/fred_data.csv to fetch the full 1997+ credit history.
    """
    out = []
    dropped = []
    for c in MACRO_FEATURE_COLS:
        if c in feat.columns and feat[c].notna().mean() >= min_nonnull_frac:
            out.append(c)
        elif c in feat.columns:
            dropped.append((c, round(float(feat[c].notna().mean()), 3)))
    if dropped:
        log.warning("Macro features dropped for low coverage (set FRED_API_KEY for full history)",
                    extra={"dropped": dropped})
    return out
