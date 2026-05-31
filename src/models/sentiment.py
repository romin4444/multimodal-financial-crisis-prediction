"""
NLP sentiment pipeline.
FinBERT (GPU FP16) + VADER lexicon baseline.
Synthetic VIX-based proxy fills coverage gaps — clearly flagged in output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import cfg
from src.logging_setup import get_logger

# NOTE: torch / transformers / vaderSentiment are imported LAZILY inside the
# functions that need them. This lets the rest of the pipeline (synthetic
# sentiment, aggregation) — and the whole test suite / CI — run without the
# heavy ~2GB NLP stack installed. Install with: pip install -e ".[nlp]"

log = get_logger("models.sentiment")

_FB_CACHE = cfg.paths.cache_dir / "finbert_scores.csv"
_FB_CKPT = cfg.paths.cache_dir / "fb_ckpt.npy"


def _resolve_device():
    """Lazily resolve torch device + batch size (requires torch)."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = cfg.finbert.batch_size_gpu if device.type == "cuda" else cfg.finbert.batch_size_cpu
    return device, batch


# ─── FinBERT ─────────────────────────────────────────────────────────────────

def run_finbert(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    GPU-accelerated FinBERT inference with caching and checkpointing.
    ProsusAI/finbert label order: index-0=positive, index-1=negative, index-2=neutral.
    Returns DataFrame with columns: date, headline, p_pos, p_neg, p_neu
    """
    if _FB_CACHE.exists():
        log.info("FinBERT scores loaded from cache")
        df = pd.read_csv(_FB_CACHE, parse_dates=["date"])
        return df

    if news_df.empty:
        log.warning("No news data — FinBERT scores unavailable")
        return pd.DataFrame(columns=["date", "headline", "p_pos", "p_neg", "p_neu"])

    device, batch_size = _resolve_device()
    log.info("Loading FinBERT", extra={"device": str(device), "batch_size": batch_size})
    tokenizer, model = _load_finbert(device)
    texts = news_df["headline"].tolist()

    all_probs: list = []
    start_idx = 0

    if _FB_CKPT.exists():
        prev = np.load(_FB_CKPT)
        all_probs.append(prev)
        start_idx = len(prev)
        log.info("Resuming FinBERT from checkpoint", extra={"resume_idx": start_idx})

    checkpoint_n = cfg.finbert.checkpoint_every_n
    for i in tqdm(range(start_idx, len(texts), batch_size), desc="FinBERT", unit="batch"):
        batch = texts[i: i + batch_size]
        try:
            probs = _infer_batch(batch, tokenizer, model, device)
        except Exception as exc:
            log.warning("FinBERT batch failed — using uniform prior", extra={"batch_start": i, "error": str(exc)})
            probs = np.full((len(batch), 3), 1 / 3, dtype=np.float32)
        all_probs.append(probs)

        if (i + BATCH_SIZE) % checkpoint_n == 0:
            np.save(_FB_CKPT, np.vstack(all_probs))

    scores = np.vstack(all_probs)
    _release_model(model)

    out = news_df[["date", "headline"]].copy()
    out["p_pos"] = scores[:, 0]
    out["p_neg"] = scores[:, 1]
    out["p_neu"] = scores[:, 2]
    out.to_csv(_FB_CACHE, index=False)

    # Clean up checkpoint after successful run
    if _FB_CKPT.exists():
        _FB_CKPT.unlink()

    log.info("FinBERT complete", extra={"headlines_scored": len(out)})
    return out


def aggregate_sentiment(scores: pd.DataFrame, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Aggregate per-headline scores to daily fear index and rolling windows.
    Forward-fills to cover weekends and holidays.
    """
    if scores.empty:
        return _zero_sentiment(trading_index)

    sc = scores.copy()
    sc["date"] = pd.to_datetime(sc["date"]).dt.tz_localize(None).dt.normalize()

    daily = (
        sc.groupby("date")
        .agg(
            fear_index=("p_neg", "mean"),
            pos_mean=("p_pos", "mean"),
            headline_count=("headline", "count"),
        )
        .reset_index()
    )
    daily["sentiment_comp"] = daily["pos_mean"] - daily["fear_index"]
    daily["panic_signal"] = (daily["fear_index"] > cfg.finbert.panic_threshold).astype(int)
    daily = daily.set_index("date").reindex(trading_index, method="ffill")

    median_fear = float(daily["fear_index"].median())
    daily["fear_index"] = daily["fear_index"].fillna(median_fear)
    daily["panic_signal"] = daily["panic_signal"].fillna(0).astype(int)
    daily["headline_count"] = daily["headline_count"].fillna(0)

    for w in [3, 7, 21]:
        daily[f"fear_{w}d"] = daily["fear_index"].rolling(w, min_periods=1).mean()

    coverage = (daily["headline_count"] > 0).mean()
    log.info("Sentiment aggregated", extra={"trading_day_coverage": round(float(coverage), 4)})
    return daily


# ─── VADER ───────────────────────────────────────────────────────────────────

def run_vader(news_df: pd.DataFrame, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """VADER lexicon baseline per Shobayo et al. (2024)."""
    if news_df.empty:
        return pd.DataFrame(index=trading_index, columns=["vader_fear", "vader_comp"])

    log.info("Running VADER baseline")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    sc = news_df.copy()
    sc["vader_neg"] = sc["headline"].apply(lambda x: analyzer.polarity_scores(str(x))["neg"])
    sc["vader_comp"] = sc["headline"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    sc["date"] = pd.to_datetime(sc["date"]).dt.tz_localize(None).dt.normalize()
    daily = (
        sc.groupby("date")
        .agg(vader_fear=("vader_neg", "mean"), vader_comp=("vader_comp", "mean"))
        .reindex(trading_index, method="ffill")
        .fillna(0)
    )
    log.info("VADER complete")
    return daily


# ─── Synthetic proxy ─────────────────────────────────────────────────────────

def build_synthetic_sentiment(feat: pd.DataFrame) -> pd.DataFrame:
    """
    VIX z-score + negative return shock → synthetic fear proxy [0, 1].
    Used to fill coverage gaps when real news is unavailable.
    Clearly flagged with is_synthetic=1 in all outputs.

    NOTE: Uses only backward-looking statistics (rolling means/stds and
    expanding min/max) — no look-ahead bias. Each value at time t depends
    only on data at times <= t.
    """
    vix = feat["vix"]
    ret = feat["log_ret"]

    vix_roll_mean = vix.rolling(252, min_periods=63).mean()
    vix_roll_std = vix.rolling(252, min_periods=63).std()
    vix_z = ((vix - vix_roll_mean) / vix_roll_std).clip(-3, 5)
    # Use expanding min/max (not global) to avoid look-ahead bias
    vix_z_min = vix_z.expanding(min_periods=63).min()
    vix_z_max = vix_z.expanding(min_periods=63).max()
    vix_scaled = (vix_z - vix_z_min) / (vix_z_max - vix_z_min + 1e-9)

    ret_roll_mean = ret.rolling(63, min_periods=21).mean()
    ret_roll_std = ret.rolling(63, min_periods=21).std()
    ret_z = (ret - ret_roll_mean) / (ret_roll_std + 1e-9)
    neg_shock = (-ret_z).clip(lower=0)
    neg_shock = neg_shock / (neg_shock.expanding(min_periods=21).max() + 1e-9)

    df = pd.DataFrame(index=feat.index)
    df["fear_index"] = (0.70 * vix_scaled + 0.30 * neg_shock).clip(0, 1).fillna(0)
    df["panic_signal"] = (df["fear_index"] > cfg.finbert.panic_threshold).astype(int)
    for w in [3, 7, 21]:
        df[f"fear_{w}d"] = df["fear_index"].rolling(w).mean()
    df["headline_count"] = 0
    df["sentiment_comp"] = 1 - df["fear_index"]
    df["is_synthetic"] = 1
    log.info("Synthetic sentiment proxy built")
    return df.fillna(0)


def merge_real_and_synthetic(
    real: pd.DataFrame, feat: pd.DataFrame, coverage_threshold: float = 0.40
) -> pd.DataFrame:
    """
    Merge real FinBERT sentiment with synthetic proxy for days with no news.
    Only fills gaps — never overwrites real sentiment.
    """
    if "headline_count" not in real.columns:
        coverage = 0.0
    else:
        coverage = float((real["headline_count"] > 0).mean())

    if coverage >= coverage_threshold:
        return real

    log.warning(
        "News coverage below threshold — merging synthetic proxy",
        extra={"coverage": round(coverage, 4), "threshold": coverage_threshold},
    )
    synth = build_synthetic_sentiment(feat)
    no_news = real.get("headline_count", pd.Series(0, index=real.index)) == 0

    merged = real.copy()
    for col in ["fear_index", "panic_signal", "fear_3d", "fear_7d", "fear_21d"]:
        if col in merged.columns and col in synth.columns:
            merged.loc[no_news, col] = synth.loc[no_news, col]

    merged["is_synthetic"] = no_news.astype(int)
    return merged


# ─── Private ─────────────────────────────────────────────────────────────────

def _load_finbert(device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.finbert.model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(cfg.finbert.model_name)
    mdl = mdl.to(device).eval()
    if device.type == "cuda":
        mdl = mdl.half()
        log.info("FinBERT: FP16 mode on GPU")
    return tok, mdl


def _infer_batch(texts: list, tokenizer, model, device) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.finbert.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.float()
        return F.softmax(logits, dim=-1).cpu().numpy()


def _release_model(model) -> None:
    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _zero_sentiment(index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = ["fear_index", "panic_signal", "headline_count", "sentiment_comp",
            "fear_3d", "fear_7d", "fear_21d"]
    return pd.DataFrame(0.0, index=index, columns=cols)
