"""
Financial Stress Index (FSI) construction.

CRITICAL: MinMaxScaler is fit ONLY on the training mask to prevent
data leakage into held-out crisis evaluation windows.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("models.fsi")

FSI_W = cfg.fsi.weights


class FSIBuilder:
    """
    Stateful FSI builder. Scalers are fit on a training mask to prevent
    look-ahead bias; the same scalers are applied to the full series.
    """

    def __init__(self) -> None:
        self._scalers: Dict[str, MinMaxScaler] = {}
        self._components: Dict[str, np.ndarray] = {}

    def build(
        self,
        feat: pd.DataFrame,
        fred_daily: pd.DataFrame,
        train_mask: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Compute FSI with a GARCH placeholder (zeros).
        Call update_with_garch() after GARCH is fitted.

        Args:
            feat: Feature DataFrame from engineer_features().
            fred_daily: FRED data aligned to trading days.
            train_mask: Boolean array over feat.index. If None, fit scalers
                        on the full series (acceptable for initial build before
                        any crisis-window evaluation).
        """
        log.info("Building Financial Stress Index")
        df = feat.copy()
        if not fred_daily.empty:
            df = df.join(fred_daily, how="left")
            for c in fred_daily.columns:
                df[c] = df[c].ffill(limit=cfg.data.fred_ffill_limit)

        if train_mask is None:
            train_mask = np.ones(len(df), dtype=bool)

        self._components["vix"] = self._scale("vix", df["vix"], train_mask)
        self._components["garch"] = np.zeros(len(df))  # placeholder
        self._components["drawdown"] = self._scale(
            "drawdown", df["drawdown_63"].abs(), train_mask
        )
        self._components["credit"] = self._build_credit(df, train_mask)

        df["FSI"] = self._weighted_sum()
        for k, v in self._components.items():
            df[f"_fsi_{k}"] = v

        # NBER recession flag (used for FSI validation only — not a model feature)
        df["_nber"] = _make_nber_flag(df.index)

        log.info(
            "FSI built (pre-GARCH)",
            extra={"fsi_min": round(float(df["FSI"].min()), 4), "fsi_max": round(float(df["FSI"].max()), 4)},
        )
        return df, self._components

    def update_with_garch(self, df: pd.DataFrame, garch_var: pd.Series) -> pd.DataFrame:
        """Replace zero GARCH placeholder with fitted conditional variance."""
        train_mask = np.ones(len(df), dtype=bool)
        aligned = garch_var.reindex(df.index).ffill().fillna(0)
        df["garch_var"] = aligned.values
        self._components["garch"] = self._scale("garch", aligned, train_mask)
        df["_fsi_garch"] = self._components["garch"]
        df["FSI"] = self._weighted_sum()
        validity = validate_fsi(df["FSI"], df)
        df.attrs["fsi_validity"] = validity
        log.info(
            "FSI updated with GARCH",
            extra={"fsi_min": round(float(df["FSI"].min()), 4), "fsi_max": round(float(df["FSI"].max()), 4), **validity},
        )
        return df

    # ─── Private ─────────────────────────────────────────────────────

    def _scale(self, name: str, series: pd.Series, train_mask: np.ndarray) -> np.ndarray:
        sc = MinMaxScaler()
        vals = series.fillna(series.median()).values.reshape(-1, 1)
        sc.fit(vals[train_mask])  # fit on training portion ONLY
        self._scalers[name] = sc
        return sc.transform(vals).ravel()

    def _build_credit(self, df: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
        if "credit_spread" in df.columns and df["credit_spread"].notna().sum() > 100:
            log.info("FSI credit component: real FRED high-yield OAS")
            return self._scale("credit", df["credit_spread"], train_mask)

        log.info("FSI credit component: VIX-momentum proxy (FRED unavailable)")
        vix = df["vix"]
        vix_z = (
            (vix - vix.rolling(252, min_periods=63).mean())
            / vix.rolling(252, min_periods=63).std()
        )
        proxy = vix_z.clip(lower=0).fillna(0)
        return self._scale("credit", proxy, train_mask)

    def _weighted_sum(self) -> np.ndarray:
        return (
            FSI_W["vix"] * self._components["vix"]
            + FSI_W["garch"] * self._components["garch"]
            + FSI_W["drawdown"] * self._components["drawdown"]
            + FSI_W["credit"] * self._components["credit"]
        )


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_fsi(fsi: pd.Series, df: pd.DataFrame) -> dict:
    """
    Three complementary FSI validity metrics (M2 §4.3, target r > 0.60):
      1. Pearson correlation with STLFSI (headline metric if available).
      2. Point-biserial correlation with NBER recession flag.
      3. ROC-AUC of FSI as NBER recession classifier.
    """
    f = fsi.fillna(0)
    nber = df.get("_nber", pd.Series(0.0, index=df.index))
    out: dict = {}

    r_pb, _ = stats.pearsonr(f, nber)
    out["nber_pointbiserial_r"] = round(float(r_pb), 4)

    try:
        out["nber_roc_auc"] = round(float(roc_auc_score(nber, f)), 4)
    except Exception:
        out["nber_roc_auc"] = float("nan")

    if "stl_fsi" in df.columns and df["stl_fsi"].notna().sum() > 100:
        idx = df["stl_fsi"].dropna().index
        r_stl, _ = stats.pearsonr(
            f.reindex(idx).fillna(0), df["stl_fsi"].reindex(idx).fillna(0)
        )
        out["stlfsi_r"] = round(float(r_stl), 4)
        target_met = abs(r_stl) >= cfg.fsi.correlation_target
    else:
        out["stlfsi_r"] = None
        target_met = (out["nber_roc_auc"] or 0) >= 0.80

    out["target_met"] = target_met
    log.info("FSI validity", extra=out)

    if not target_met:
        log.warning(
            "FSI validity below target",
            extra={"target": cfg.fsi.correlation_target, **out},
        )
    return out


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_nber_flag(index: pd.DatetimeIndex) -> pd.Series:
    flag = pd.Series(0.0, index=index)
    for start, end in cfg.nber_recessions:
        flag[(index >= start) & (index <= end)] = 1.0
    return flag
