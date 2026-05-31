"""
Financial news data loading.
Auto-detects date + headline columns from any CSV directory — no hardcoded paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("data.news")

_CACHE_PATH = cfg.paths.cache_dir / "news_raw.csv"

_DATE_KEYS = frozenset({
    "date", "datetime", "time", "published", "publisheddate",
    "publish_date", "article_date", "release_date", "created_at",
    "timestamp", "posted_date", "news_date", "datetime_utc", "pubdate",
})
_TEXT_KEYS = frozenset({
    "headline", "title", "news", "text", "content", "article",
    "body", "story", "description", "summary", "headline_text",
    "news_headline", "article_headline", "titles",
})


def load_news(news_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load financial news headlines from a directory of CSVs.
    Falls back to /kaggle/input if news_dir is not provided.
    Returns DataFrame with columns ['date', 'headline'].
    """
    if _CACHE_PATH.exists():
        log.info("News loaded from cache", extra={"path": str(_CACHE_PATH)})
        df = pd.read_csv(_CACHE_PATH, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"])

    search_dir = _resolve_news_dir(news_dir)
    if search_dir is None:
        log.warning("No news directory found — pipeline will use synthetic sentiment proxy")
        return _empty_news()

    csv_files = [f for f in search_dir.rglob("*.csv") if f.stat().st_size > 10_000]
    log.info("Scanning CSVs for news", extra={"directory": str(search_dir), "files": len(csv_files)})

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        frame = _try_load_csv(csv_path)
        if frame is not None:
            frames.append(frame)
            log.info("News CSV loaded", extra={"file": csv_path.name, "rows": len(frame)})

    if not frames:
        log.warning("No valid news CSVs found — using synthetic sentiment proxy")
        return _empty_news()

    news = pd.concat(frames, ignore_index=True)
    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    news = _clean_news(news)

    news.to_csv(_CACHE_PATH, index=False)
    log.info(
        "News data ready",
        extra={
            "rows": len(news),
            "date_min": str(news["date"].min().date()),
            "date_max": str(news["date"].max().date()),
        },
    )
    return news


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_news_dir(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None and explicit.exists():
        return explicit
    # Check environment variable
    env_dir = os.environ.get("NEWS_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    # Kaggle fallback
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        return kaggle
    return None


def _try_load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        # Probe first 50 rows to detect schema cheaply
        probe = pd.read_csv(
            path, low_memory=False, nrows=50,
            encoding="utf-8", encoding_errors="replace",
        )
        col_map = {c: c.lower().replace(" ", "_").strip() for c in probe.columns}
        probe = probe.rename(columns=col_map)

        date_col = next((c for c in probe.columns if c in _DATE_KEYS), None)
        text_col = next((c for c in probe.columns if c in _TEXT_KEYS), None)
        if not (date_col and text_col):
            return None

        # Reject price-only CSVs: genuine headlines are long strings
        avg_len = probe[text_col].astype(str).str.len().mean()
        if avg_len < cfg.data.news_min_headline_len:
            return None

        full = pd.read_csv(
            path, low_memory=False, encoding="utf-8", encoding_errors="replace"
        )
        full = full.rename(columns={c: c.lower().replace(" ", "_").strip() for c in full.columns})
        df = full[[date_col, text_col]].rename(columns={date_col: "date", text_col: "headline"})
        df = df.dropna()
        df["headline"] = df["headline"].astype(str).str.strip()
        df = df[df["headline"].str.len() > 10]
        return df if len(df) > 0 else None

    except Exception as exc:
        log.debug("Skipping CSV", extra={"file": path.name, "reason": str(exc)})
        return None


def _clean_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date"])
    # Strip timezone info for consistency
    try:
        df["date"] = df["date"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        df["date"] = pd.to_datetime(
            df["date"].astype(str).str.slice(0, 19), errors="coerce"
        )
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["headline"]).sort_values("date")
    max_h = cfg.data.max_headlines
    if len(df) > max_h:
        df = df.tail(max_h)  # keep newest to favour recent crisis coverage
        log.info("Headlines capped", extra={"kept": max_h})
    return df.reset_index(drop=True)


def _empty_news() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "headline"])
