"""
FRED economic data acquisition.
Uses FRED REST API directly — no hard dependency on kaggle_secrets.
API key loaded from environment variable FRED_API_KEY.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("data.fred")

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_CACHE_PATH = cfg.paths.cache_dir / "fred_data.csv"


def load_fred_key() -> str:
    """Resolve FRED API key from environment. Returns empty string if not set."""
    # Priority: FRED_API_KEY > KAGGLE_SECRET_FRED_API_KEY
    key = os.environ.get("FRED_API_KEY", "") or os.environ.get("KAGGLE_SECRET_FRED_API_KEY", "")
    if key:
        return key.strip()

    # Kaggle Secrets fallback — only imported if running on Kaggle
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import]
        secret = UserSecretsClient().get_secret("KAGGLE_SECRET_FRED_API_KEY")
        return secret.strip() if secret else ""
    except Exception:
        return ""


def download_fred(api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Download FRED indicator series and cache locally.
    Returns empty DataFrame if no API key is available — pipeline degrades gracefully.
    """
    if _CACHE_PATH.exists():
        df = pd.read_csv(_CACHE_PATH, index_col=0, parse_dates=True)
        log.info("FRED data loaded from cache", extra={"rows": len(df), "cols": list(df.columns)})
        return df

    key = api_key or load_fred_key()
    if not key:
        log.warning(
            "FRED API key not found. Set environment variable FRED_API_KEY. "
            "FSI will use VIX proxy for credit-spread component."
        )
        return pd.DataFrame()

    log.info("Fetching FRED series", extra={"series": list(cfg.data.fred_series.keys())})
    series: Dict[str, pd.Series] = {}

    for sid, col_name in cfg.data.fred_series.items():
        fetched = _fetch_series(sid, key)
        if fetched is not None:
            series[col_name] = fetched

    if not series:
        log.error("No FRED series could be fetched")
        return pd.DataFrame()

    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index)

    # Consolidate STLFSI: prefer v4 (STLFSI4), fall back to legacy v2
    if "stl_fsi" not in df.columns and "stl_fsi_v2" in df.columns:
        df["stl_fsi"] = df["stl_fsi_v2"]

    df.to_csv(_CACHE_PATH)
    log.info("FRED data cached", extra={"rows": len(df), "path": str(_CACHE_PATH)})
    return df


def align_fred_to_trading_days(
    fred_df: pd.DataFrame, trading_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Align FRED (weekly/monthly) data to daily trading calendar.
    Uses forward-fill with a configurable limit to avoid propagating stale values.
    """
    if fred_df.empty:
        return pd.DataFrame(index=trading_index)

    out = pd.DataFrame(index=trading_index)
    limit = cfg.data.fred_ffill_limit  # cap stale-value propagation

    for col in fred_df.columns:
        series = fred_df[col].dropna()
        if len(series) >= 5:
            out[col] = series.reindex(trading_index, method="ffill", limit=limit)

    missing_pct = out.isna().mean().to_dict()
    for col, pct in missing_pct.items():
        if pct > 0.1:
            log.warning(
                "FRED series has >10%% missing after alignment",
                extra={"column": col, "missing_pct": round(pct, 4)},
            )

    return out


# ─── Internal ────────────────────────────────────────────────────────────────

def _fetch_series(series_id: str, api_key: str) -> Optional[pd.Series]:
    try:
        response = requests.get(
            _FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": cfg.data.start_date,
                "observation_end": cfg.data.end_date,
            },
            timeout=30,
        )
        response.raise_for_status()
        observations = response.json().get("observations", [])

        if not observations:
            log.warning("No observations returned", extra={"series_id": series_id})
            return None

        values = {
            pd.to_datetime(o["date"]): (
                np.nan if o["value"] in (".", "") else float(o["value"])
            )
            for o in observations
        }
        series = pd.Series(values).sort_index()
        valid = series.notna().sum()

        if valid < 5:
            log.warning("Too few valid observations", extra={"series_id": series_id, "valid": valid})
            return None

        log.info("FRED series fetched", extra={"series_id": series_id, "valid_obs": valid})
        return series

    except requests.RequestException as exc:
        log.error("FRED HTTP request failed", extra={"series_id": series_id, "error": str(exc)})
        return None
    except Exception as exc:
        log.error("FRED fetch error", extra={"series_id": series_id, "error": str(exc)})
        return None
