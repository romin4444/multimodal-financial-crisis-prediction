"""
v3 — VIX-orthogonal option-surface features.

THE PROBLEM (established by Run 1-4 in EDGE_AND_MULTIMODAL_RESULTS.md):
    Every feature family tried so far (realized vol, drawdown, regime, credit,
    yield curve, oil, cross-asset corr, TDA, FinBERT sentiment) FAILS to beat a
    VIX threshold out-of-sample, because each is largely COLLINEAR with VIX. On a
    ~4%-positive class, collinear features add variance, not signal.

THE ONLY WAY TO BEAT VIX:
    Add information the option-implied vol LEVEL does not already contain. The
    most literature-backed candidate is the **Variance Risk Premium** (VRP):

        VRP_t = IV_t^2 - RV_t^2      (implied variance minus realized variance)

    VRP carries forward-predictive content distinct from the IV level
    (Bollerslev, Tauchen & Zhou 2009, RFS). Crucially it needs NO new data —
    both terms are already in the pipeline (VIX and realized vol).

    Two further option-surface signals are genuinely orthogonal but need a feed:
      - VIX term-structure slope (VIX9D / VIX / VIX3M): inverts before stress.
      - CBOE SKEW: OTM put demand = crash-risk pricing distinct from the level.
    They are built here when their inputs are supplied, else cleanly skipped.

ORTHOGONALIZE-THEN-TEST:
    A feature can only beat VIX if it has a VIX-orthogonal component. Each raw
    signal is residualised against VIX (and a VIX lag) with an OLS fit on a
    TRAIN-ONLY head slice (no look-ahead), and the residual is exposed as
    `<name>_resid`. The decisive experiment then compares, out-of-sample:
        VIX-only   vs   VIX + orthogonal-residuals.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

VIX_ORTHO_BASE = ["vrp", "vrp_ratio", "vol_of_vol"]


# ─── core signals (no extra data needed) ──────────────────────────────────────

def variance_risk_premium(vix: pd.Series, realized_vol_ann: pd.Series) -> pd.Series:
    """
    VRP = (VIX/100)^2 - realized_variance.

    `vix` is in annualised vol points (e.g. 20.0 = 20%); `realized_vol_ann` is
    annualised realized vol in decimal (e.g. 0.18). Both are converted to
    annualised *variance* in decimal so the difference is well-posed. Uses only
    contemporaneous implied and TRAILING realized vol -> causal.
    """
    iv_var = (vix.astype(float) / 100.0) ** 2
    rv_var = realized_vol_ann.astype(float) ** 2
    return (iv_var - rv_var).rename("vrp")


def vrp_ratio(vix: pd.Series, realized_vol_ann: pd.Series) -> pd.Series:
    """Implied/realized variance ratio — scale-free VRP twin. >1 normally."""
    iv_var = (vix.astype(float) / 100.0) ** 2
    rv_var = (realized_vol_ann.astype(float) ** 2).replace(0, np.nan)
    return (iv_var / rv_var).rename("vrp_ratio")


def vol_of_vol(vix: pd.Series, window: int = 21) -> pd.Series:
    """Rolling std of daily VIX changes — vol-of-vol, partly orthogonal to level."""
    return vix.astype(float).diff().rolling(window).std().rename("vol_of_vol")


# ─── optional option-surface signals (need a data feed) ───────────────────────

def vix_term_structure(
    vix: pd.Series,
    vix9d: Optional[pd.Series] = None,
    vix3m: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Term-structure slopes. Backwardation (short > long) precedes stress.
    Returns whatever can be built from the supplied maturities (may be empty).
    """
    out = pd.DataFrame(index=vix.index)
    v = vix.astype(float)
    if vix9d is not None:
        out["ts_9d_1m"] = (vix9d.astype(float) / v) - 1.0       # >0 = backwardation
    if vix3m is not None:
        out["ts_1m_3m"] = (v / vix3m.astype(float)) - 1.0       # >0 = backwardation
    return out


def skew_feature(skew: Optional[pd.Series]) -> pd.DataFrame:
    """CBOE SKEW (≈100-150). Higher = more OTM put (crash) demand."""
    out = pd.DataFrame(index=skew.index) if skew is not None else pd.DataFrame()
    if skew is not None:
        s = skew.astype(float)
        out["skew_level"] = s - 100.0
        out["skew_chg_21"] = s.diff(21)
    return out


# ─── orthogonalize-then-test machinery ────────────────────────────────────────

def residualize_on_vix(
    feature: pd.Series,
    vix: pd.Series,
    train_frac: float = 0.5,
    use_lag: bool = True,
) -> pd.Series:
    """
    Return the component of `feature` orthogonal to VIX (and a 1-day VIX lag),
    via an OLS fit on the first `train_frac` of the sample ONLY (leakage-free),
    applied to the whole series. The residual is the part a VIX rule cannot see.
    """
    df = pd.DataFrame({"f": feature.astype(float), "vix": vix.astype(float)})
    if use_lag:
        df["vix_lag"] = df["vix"].shift(1)
    df = df.dropna()
    if len(df) < 200:
        return (feature - feature.mean()).rename(f"{feature.name}_resid")

    cut = int(len(df) * train_frac)
    cols = ["vix", "vix_lag"] if use_lag else ["vix"]
    Xtr = np.column_stack([np.ones(cut)] + [df[c].to_numpy()[:cut] for c in cols])
    ytr = df["f"].to_numpy()[:cut]
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

    Xall = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    resid = df["f"].to_numpy() - Xall @ beta
    return pd.Series(resid, index=df.index, name=f"{feature.name}_resid")


def orthogonality_report(feature: pd.Series, vix: pd.Series) -> Dict[str, float]:
    """How much of `feature` is NOT explained by VIX? (1 - R^2 of feature~vix)."""
    df = pd.DataFrame({"f": feature.astype(float), "vix": vix.astype(float)}).dropna()
    if len(df) < 50:
        return {"corr_with_vix": float("nan"), "orthogonal_frac": float("nan")}
    corr = float(np.corrcoef(df["f"], df["vix"])[0, 1])
    return {
        "corr_with_vix": round(corr, 4),
        "orthogonal_frac": round(1.0 - corr ** 2, 4),  # share of variance VIX can't explain
    }


def build_vix_orthogonal(
    feat: pd.DataFrame,
    vix9d: Optional[pd.Series] = None,
    vix3m: Optional[pd.Series] = None,
    skew: Optional[pd.Series] = None,
    realized_col: str = "vol_21d",
    train_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Assemble all available VIX-orthogonal features on `feat`'s index.

    Requires `feat` to contain 'vix' and a realized-vol column (`realized_col`).
    Adds raw signals AND their VIX-residualised twins (`*_resid`).
    """
    if "vix" not in feat.columns:
        raise ValueError("feat must contain a 'vix' column")
    vix = feat["vix"]
    rv = feat[realized_col]

    out = pd.DataFrame(index=feat.index)
    out["vrp"] = variance_risk_premium(vix, rv)
    out["vrp_ratio"] = vrp_ratio(vix, rv)
    out["vol_of_vol"] = vol_of_vol(vix)

    ts = vix_term_structure(vix, vix9d, vix3m)
    for c in ts.columns:
        out[c] = ts[c]
    sk = skew_feature(skew)
    for c in sk.columns:
        out[c] = sk[c]

    # orthogonalised residuals (leakage-free, train-head OLS)
    for c in list(out.columns):
        out[f"{c}_resid"] = residualize_on_vix(out[c], vix, train_frac=train_frac)

    return out


def available_ortho_cols(ortho: pd.DataFrame, min_nonnull_frac: float = 0.7) -> List[str]:
    """Columns with enough coverage to be usable (drops missing option-feed signals)."""
    return [c for c in ortho.columns if ortho[c].notna().mean() >= min_nonnull_frac]
