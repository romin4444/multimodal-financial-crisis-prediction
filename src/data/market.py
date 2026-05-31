"""
Market data acquisition — yfinance with retry logic and CSV fallback.
Portable: no Kaggle-specific paths or secrets APIs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("data.market")

# ─── Public API ──────────────────────────────────────────────────────────────

def download_all_market() -> Dict[str, pd.DataFrame]:
    """Download S&P 500, VIX, and individual stock tickers."""
    log.info("Downloading market data", extra={"tickers": [cfg.data.index_ticker, cfg.data.vix_ticker] + cfg.data.stock_tickers})
    result: Dict[str, pd.DataFrame] = {
        "sp500": _download_ticker(cfg.data.index_ticker),
        "vix": _download_ticker(cfg.data.vix_ticker),
    }
    for ticker in cfg.data.stock_tickers:
        result[ticker.lower()] = _download_ticker(ticker)
    log.info("Market data ready", extra={"series": list(result.keys())})
    return result


# ─── Internal helpers ────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("^", "").replace("/", "-")
    return cfg.paths.cache_dir / f"mkt_{safe}.csv"


def _download_ticker(ticker: str) -> pd.DataFrame:
    """Download or retrieve cached OHLCV data for a single ticker."""
    cache = _cache_path(ticker)
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        log.debug("Ticker loaded from cache", extra={"ticker": ticker, "rows": len(df)})
        return df

    df = _fetch_yfinance(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError(
            f"Failed to obtain data for {ticker}. "
            "Ensure internet access is available or attach an OHLCV CSV."
        )
    df.to_csv(cache)
    log.info("Ticker downloaded and cached", extra={"ticker": ticker, "rows": len(df)})
    return df


def _fetch_yfinance(ticker: str) -> Optional[pd.DataFrame]:
    """Attempt yfinance download with exponential backoff retries."""
    retries = cfg.data.yfinance_retries
    backoff = cfg.data.yfinance_retry_backoff

    for attempt in range(retries):
        try:
            log.debug("yfinance fetch attempt", extra={"ticker": ticker, "attempt": attempt + 1})
            df = yf.download(
                ticker,
                start=cfg.data.start_date,
                end=cfg.data.end_date,
                auto_adjust=True,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and not df.empty and "Close" in df.columns:
                return df
        except Exception as exc:
            wait = backoff * (attempt + 1)
            log.warning(
                "yfinance fetch failed",
                extra={"ticker": ticker, "attempt": attempt + 1, "error": str(exc), "retry_in_s": wait},
            )
            time.sleep(wait)

    return None
