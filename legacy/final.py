#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  MBAI 5600G  |  Group 13  |  Jeya Surya Balaji & Keertan Patel  ║
║  Multimodal Financial Crisis Prediction — Kaggle Production v2   ║
╚══════════════════════════════════════════════════════════════════╝

KAGGLE SETUP (must do before running):
  1. Settings → Accelerator → GPU T4 x2          (makes FinBERT ~10× faster)
  2. Settings → Internet → ON                     (yfinance / HuggingFace)
  3. Add-ons → Secrets → KAGGLE_SECRET_FRED_API_KEY  (free at fred.stlouisfed.org)
  4. Add dataset: search "financial news stock price integration" → attach as input
  5. (Optional) Add: "daily financial news 6000 stocks" as second input

MODELS IMPLEMENTED:
  ┌─ Quantitative Pipeline (Person A) ──────────────────────────────┐
  │  ARMA-GARCH(1,1)/GJR-GARCH/EGARCH  → BIC model selection       │
  │  Financial Stress Index (FSI)       → 4-component composite     │
  │  Gaussian HMM  n∈{2,3,4}           → 50 seeds, BIC selection    │
  └─────────────────────────────────────────────────────────────────┘
  ┌─ NLP Pipeline (Person B) ───────────────────────────────────────┐
  │  FinBERT (ProsusAI/finbert)         → GPU FP16, batch-128       │
  │  VADER (lexicon baseline)           → Shobayo 2024 replication  │
  │  Synthetic VIX-proxy                → gap-fill 2020/2022        │
  └─────────────────────────────────────────────────────────────────┘
  ┌─ Integration & Validation ──────────────────────────────────────┐
  │  Lead-lag cross-correlation ±30d    → bootstrap 1000 CI         │
  │  Granger causality (Bollen 2011)                                │
  │  Logistic Regression + Random Forest fusion                     │
  │  SHAP (LinearExplainer + TreeExplainer)   per crisis window     │
  │  Wang et al. 2025 HMM-only baseline  replication                │
  └─────────────────────────────────────────────────────────────────┘

OUTPUTS:  /kaggle/working/outputs/
  01_regime_timeline.png   02_sentiment_vs_fsi.png  03_lead_lag.png
  04_shap_by_crisis.png    05_hmm_selection.png     06_garch_all.png
  07_fusion_eval.png       08_research_comparison.png
  integration_master.csv   models/  (pickle files)
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 1 — INSTALL (run once per Kaggle session)
# ═══════════════════════════════════════════════════════════════════
import subprocess, sys

_PKGS = [
    "yfinance>=0.2.36", "fredapi>=0.5.2", "hmmlearn>=0.3.3",
    "arch>=6.3.0", "transformers>=4.38.0", "shap>=0.44.0",
    "scipy>=1.11.0", "scikit-learn>=1.4.0", "statsmodels>=0.14.0",
    "vaderSentiment>=3.3.2", "tqdm>=4.66.0",
    "plotly>=5.18.0", "kaleido>=0.2.1",
]
for pkg in _PKGS:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
print("✅ All packages ready")

# ═══════════════════════════════════════════════════════════════════
# CELL 2 — IMPORTS
# ═══════════════════════════════════════════════════════════════════
import os, warnings, pickle, json, logging, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
pd.set_option("display.float_format", "{:.4f}".format)

from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, classification_report,
)
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.diagnostic import het_arch
import statsmodels.api as sm

import yfinance as yf
try:
    from fredapi import Fred
    _FRED_OK = True
except ImportError:
    _FRED_OK = False

from arch import arch_model
from hmmlearn.hmm import GaussianHMM

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm.auto import tqdm

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════
# CELL 3 — CONFIGURATION  (edit here only)
# ═══════════════════════════════════════════════════════════════════

# ── Reproducibility ────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU  : {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Directories ────────────────────────────────────────────────────
CACHE_DIR   = Path("/kaggle/working/cache")
OUTPUT_DIR  = Path("/kaggle/working/outputs")
MODEL_DIR   = Path("/kaggle/working/outputs/models")
for d in [CACHE_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Date / Tickers ─────────────────────────────────────────────────
START_DATE    = "1990-01-01"
END_DATE      = "2024-12-31"
INDEX_TICKER  = "^GSPC"
VIX_TICKER    = "^VIX"
STOCK_TICKERS = ["AAPL", "JPM", "XOM", "GS"]    # Tech / Finance / Energy / Finance-bellwether

# Offline-fallback datasets (attached as Kaggle inputs). If yfinance is rate-
# limited or blocked, the loader recovers OHLCV from these attached CSVs.
# Goldman Sachs (anadiskt/goldman-sachs-gs-stock-data-19992026) covers 1999-2026.
TICKER_FALLBACK_HINTS = {
    "GS": ["goldman", "gs_stock", "gs-stock"],
}

# Optional emerging-market robustness appendix (out of core S&P 500 scope).
# khuong11/vn-quant-master-db-2014-042024 — runs ONLY if compatible OHLCV found.
VN_DATASET_HINTS = ["vn-quant", "vn_quant", "vnquant", "vietnam"]

# ── FRED ───────────────────────────────────────────────────────────
# Loaded from Kaggle Secrets (KAGGLE_SECRET_FRED_API_KEY) or env var.
def _load_fred_key() -> str:
    k = os.environ.get("KAGGLE_SECRET_FRED_API_KEY", "")
    if k:
        return k.strip()
    try:                                              # Kaggle Secrets API
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("KAGGLE_SECRET_FRED_API_KEY").strip()
    except Exception:
        return ""

FRED_KEY = _load_fred_key()
FRED_SERIES = {
    "FEDFUNDS":     "fed_funds",
    "T10Y2Y":       "yield_spread",
    "BAMLH0A0HYM2": "credit_spread",     # High-Yield OAS — key 2008 stress signal
    "STLFSI4":      "stl_fsi",           # St. Louis Fed Financial Stress Index (4th release)
    "STLFSI2":      "stl_fsi_v2",        # legacy fallback series
    "DCOILWTICO":   "oil_price",
}

# ── FSI weights (M2, Section 4.1) ──────────────────────────────────
FSI_W = {"vix": 0.30, "garch": 0.30, "drawdown": 0.20, "credit": 0.20}

# ── GARCH specs to compare (Huang & Luo 2024) ──────────────────────
GARCH_SPECS = [
    {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "label": "GARCH(1,1)"},
    {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "label": "GJR-GARCH(1,1)"},
    {"vol": "EGARCH","p": 1, "o": 1, "q": 1, "label": "EGARCH(1,1)"},
]

# ── HMM ────────────────────────────────────────────────────────────
HMM_N_LIST = [2, 3, 4]  # fit all three to report the BIC profile
HMM_FORCE_N = 3         # CANONICAL regime model = 3 states (stable/volatile/crisis).
#   Rationale: with N≈8,700 daily obs the parameter penalty in BIC is dwarfed by
#   the likelihood gain, so BIC decreases monotonically with state count and would
#   keep selecting the largest n. Following Ang & Timmermann (2012), who show 2-4
#   state HMMs are standard for equity regimes, we FIX n=3 for interpretability —
#   it maps cleanly to the stable / volatile / crisis taxonomy used throughout the
#   project and keeps the crisis state (highest volatility) unambiguous. The full
#   BIC/LL profile across n∈{2,3,4} is still reported (figure 05) for transparency.
HMM_N_INIT = 50         # random seeds — higher = more reliable EM convergence
HMM_N_ITER = 200        # max EM steps per seed

# ── FinBERT ────────────────────────────────────────────────────────
FINBERT_MODEL  = "ProsusAI/finbert"
FINBERT_BATCH  = 128 if DEVICE.type == "cuda" else 32
FINBERT_MAXLEN = 128
PANIC_THR      = 0.40   # P(negative) > 40% → panic day
MAX_HEADLINES  = 400_000  # cap to bound FinBERT runtime (deduped, newest kept)

# ── Lead-lag ───────────────────────────────────────────────────────
MAX_LAG     = 30         # ±30 trading days
BOOT_N      = 1000       # bootstrap iterations

# ── Fusion ─────────────────────────────────────────────────────────
PRED_HORIZON = 5         # trading days ahead for target construction

# ── Crisis validation windows (M2 Section 4.5) ─────────────────────
CRISIS_WINDOWS = {
    "GFC_2008":       ("2008-09-01", "2009-03-31"),
    "COVID_2020":     ("2020-02-19", "2020-03-23"),
    "Inflation_2022": ("2022-01-01", "2022-10-31"),
}

# NBER US recessions (hardcoded for FSI validation target r > 0.60)
NBER = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

# ── Targets (M2 Section 4.3) ───────────────────────────────────────
FSI_CORR_TARGET  = 0.60
FUSION_F1_TARGET = 0.70

# ── Palette ────────────────────────────────────────────────────────
C = {
    "stable":    "#2ECC71", "volatile": "#F39C12",
    "crisis":    "#E74C3C", "sentiment":"#3498DB",
    "fsi":       "#9B59B6", "garch":    "#E67E22",
    "vader":     "#95A5A6", "price":    "#1ABC9C",
}
sns.set_theme(style="whitegrid")
print("✅ Configuration complete  |  Device:", DEVICE)

# ═══════════════════════════════════════════════════════════════════
# CELL 4 — DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════

def _cp(name: str) -> Path:
    """Cache path helper."""
    return CACHE_DIR / f"{name}.csv"


def _find_attached_csv(ticker: str) -> Optional[pd.DataFrame]:
    """Recover OHLCV for a ticker from any attached Kaggle dataset CSV."""
    inp = Path("/kaggle/input")
    if not inp.exists():
        return None
    hints = [ticker.lower()] + TICKER_FALLBACK_HINTS.get(ticker.upper(), [])
    for csv in inp.rglob("*.csv"):
        name = csv.name.lower()
        path = str(csv).lower()
        if not any(h in name or h in path for h in hints):
            continue
        try:
            df = pd.read_csv(csv)
            cols = {c.lower().strip(): c for c in df.columns}
            dcol = next((cols[c] for c in cols if c in
                         {"date", "datetime", "time", "timestamp"}), None)
            ccol = next((cols[c] for c in cols if c in
                         {"close", "adj close", "adj_close", "closing price", "price"}), None)
            if not (dcol and ccol):
                continue
            out = pd.DataFrame()
            out.index = pd.to_datetime(df[dcol], errors="coerce")
            for std, keys in {"Open": {"open"}, "High": {"high"}, "Low": {"low"},
                              "Close": {"close", "adj close", "adj_close", "price"},
                              "Volume": {"volume", "vol"}}.items():
                src = next((cols[c] for c in cols if c in keys), None)
                if src is not None:
                    out[std] = pd.to_numeric(df[src], errors="coerce")
            out = out[~out.index.isna()].sort_index()
            out = out[~out.index.duplicated(keep="last")]
            if "Close" in out and out["Close"].notna().sum() > 200:
                logger.info(f"  ↪ {ticker}: recovered {len(out):,} rows from {csv.name}")
                return out
        except Exception:
            continue
    return None


def _dl_ticker(ticker: str) -> pd.DataFrame:
    safe = ticker.replace("^", "").replace("/", "-")
    p = CACHE_DIR / f"mkt_{safe}.csv"
    if p.exists():
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    df = pd.DataFrame()
    for attempt in range(3):                       # retry yfinance (flaky on Kaggle)
        try:
            logger.info(f"  Downloading {ticker} … (try {attempt+1})")
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and not df.empty and "Close" in df.columns:
                break
        except Exception as e:
            logger.warning(f"    yfinance error: {e}")
            time.sleep(2 * (attempt + 1))

    if df is None or df.empty or "Close" not in df.columns:
        logger.warning(f"  yfinance unavailable for {ticker} — trying attached CSV")
        fb = _find_attached_csv(ticker)
        if fb is not None:
            df = fb
        else:
            raise RuntimeError(
                f"Could not obtain {ticker} from yfinance OR attached datasets. "
                f"Ensure Internet=ON, or attach an OHLCV dataset for {ticker}.")
    df.to_csv(p)
    return df


def download_all_market() -> Dict[str, pd.DataFrame]:
    logger.info("[DATA] Market tickers …")
    data = {"sp500": _dl_ticker(INDEX_TICKER),
            "vix":   _dl_ticker(VIX_TICKER)}
    for t in STOCK_TICKERS:
        data[t.lower()] = _dl_ticker(t)
    logger.info(f"  Loaded: {list(data.keys())}")
    return data


def download_fred() -> pd.DataFrame:
    """
    Fetch FRED series via the public REST API using the Kaggle-Secrets key.
    Uses `requests` directly (no fredapi dependency) for maximum robustness.
    """
    p = _cp("fred_data")
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    if not FRED_KEY:
        logger.warning("[DATA] No FRED key (Add-ons → Secrets → "
                       "KAGGLE_SECRET_FRED_API_KEY) — FSI uses VIX proxy")
        return pd.DataFrame()

    import requests
    logger.info("[DATA] FRED series via REST …")
    base = "https://api.stlouisfed.org/fred/series/observations"
    series = {}
    for sid, col in FRED_SERIES.items():
        try:
            r = requests.get(base, params={
                "series_id": sid, "api_key": FRED_KEY, "file_type": "json",
                "observation_start": START_DATE, "observation_end": END_DATE,
            }, timeout=30)
            obs = r.json().get("observations", [])
            if not obs:
                logger.warning(f"  ✗ {sid}: no observations")
                continue
            s = pd.Series(
                {pd.to_datetime(o["date"]):
                 (np.nan if o["value"] in (".", "") else float(o["value"]))
                 for o in obs}).sort_index()
            if s.notna().sum() >= 5:
                series[col] = s
                logger.info(f"  ✓ {sid} ({s.notna().sum():,} obs)")
        except Exception as e:
            logger.warning(f"  ✗ {sid}: {e}")
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index)
    # Consolidate STLFSI: prefer v4, fall back to legacy v2
    if "stl_fsi" not in df.columns and "stl_fsi_v2" in df.columns:
        df["stl_fsi"] = df["stl_fsi_v2"]
    df.to_csv(p)
    return df


def load_news() -> pd.DataFrame:
    """
    Robustly load financial-news headlines from ANY attached Kaggle dataset by
    recursively scanning /kaggle/input (no hard-coded paths). Auto-detects the
    date + headline columns, filters out price-only CSVs, dedupes, and caps the
    total to MAX_HEADLINES (newest kept) to bound FinBERT runtime.
    """
    p = _cp("news_raw")
    if p.exists():
        logger.info("[DATA] News from cache …")
        df = pd.read_csv(p, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"])

    DATE_KEYS = {"date", "datetime", "time", "published", "publisheddate",
                 "publish_date", "article_date", "release_date", "created_at",
                 "timestamp", "posted_date", "news_date", "datetime_utc", "pubdate"}
    TEXT_KEYS = {"headline", "title", "news", "text", "content", "article",
                 "body", "story", "description", "summary", "headline_text",
                 "news_headline", "article_headline", "titles"}

    inp = Path("/kaggle/input")
    if not inp.exists():
        logger.warning("[DATA] /kaggle/input not found — using synthetic proxy")
        return pd.DataFrame(columns=["date", "headline"])

    all_csvs = [c for c in inp.rglob("*.csv") if c.stat().st_size > 10_000]
    logger.info(f"[DATA] Scanning {len(all_csvs)} CSVs in /kaggle/input for news …")
    frames = []
    for csv in all_csvs:
        try:
            head = pd.read_csv(csv, low_memory=False, nrows=50,
                               encoding="utf-8", encoding_errors="replace")
            cl = {c: c.lower().replace(" ", "_").strip() for c in head.columns}
            head = head.rename(columns=cl)
            dc = next((c for c in head.columns if c in DATE_KEYS), None)
            tc = next((c for c in head.columns if c in TEXT_KEYS), None)
            if not (dc and tc):
                continue
            # Reject price-only CSVs masquerading via a 'price'/'close' text col:
            avg_len = head[tc].astype(str).str.len().mean()
            if avg_len < 15:                      # real headlines are long strings
                continue
            full = pd.read_csv(csv, low_memory=False, encoding="utf-8",
                               encoding_errors="replace")
            full = full.rename(columns={c: c.lower().replace(" ", "_").strip()
                                        for c in full.columns})
            full = full[[dc, tc]].rename(columns={dc: "date", tc: "headline"})
            full = full.dropna()
            full["headline"] = full["headline"].astype(str).str.strip()
            full = full[full["headline"].str.len() > 10]
            if len(full):
                frames.append(full)
                logger.info(f"  ✓ {csv.relative_to(inp)}: {len(full):,} rows")
        except Exception as e:
            logger.debug(f"  skip {csv.name}: {e}")

    if not frames:
        logger.warning("[DATA] No news CSV detected — using VIX-based synthetic proxy")
        return pd.DataFrame(columns=["date", "headline"])

    news = pd.concat(frames, ignore_index=True)
    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    news = news.dropna(subset=["date"])
    try:
        news["date"] = news["date"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        news["date"] = pd.to_datetime(news["date"].astype(str).str.slice(0, 19),
                                      errors="coerce")
    news = news.dropna(subset=["date"])
    news = news.drop_duplicates(subset=["headline"]).sort_values("date")
    if len(news) > MAX_HEADLINES:                 # keep newest to favour 2020/2022 coverage
        news = news.tail(MAX_HEADLINES)
        logger.info(f"  Capped to newest {MAX_HEADLINES:,} headlines")
    news = news.reset_index(drop=True)
    news.to_csv(p, index=False)
    logger.info(f"[DATA] News total: {len(news):,} rows | "
                f"{news['date'].min().date()} → {news['date'].max().date()}")
    return news

# ═══════════════════════════════════════════════════════════════════
# CELL 5 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def engineer_features(sp500: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all price-based features. Runs ADF + ARCH-LM diagnostics.
    All column names are snake_case and consistent throughout the pipeline.
    """
    logger.info("[FEAT] Engineering features …")
    df = pd.DataFrame(index=sp500.index)
    df["close"]  = sp500["Close"]
    df["volume"] = sp500["Volume"]
    df["vix"]    = vix["Close"].reindex(df.index).ffill()

    # ── Log returns ──────────────────────────────────────────────
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # ── Rolling volatility (annualised) ──────────────────────────
    for w in [5, 21, 63, 126]:
        df[f"vol_{w}d"] = df["log_ret"].rolling(w).std() * np.sqrt(252)

    # ── Max drawdown over 63-day window ──────────────────────────
    df["drawdown_63"] = (
        df["close"]
        .rolling(63)
        .apply(lambda x: (x[-1] - x.max()) / x.max() if x.max() != 0 else 0,
               raw=True)
    )

    # ── VIX features ─────────────────────────────────────────────
    df["vix_chg"]   = df["vix"].pct_change()
    df["vix_ma21"]  = df["vix"].rolling(21).mean()
    df["vix_spike"] = (
        df["vix"] > df["vix"].rolling(63).mean() + 2 * df["vix"].rolling(63).std()
    ).astype(int)

    # ── Price momentum ────────────────────────────────────────────
    for d in [5, 21, 63]:
        df[f"mom_{d}d"] = df["close"].pct_change(d)

    # ── Volume ratio ──────────────────────────────────────────────
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(21).mean()

    # Initialise GARCH variance placeholder (updated after GARCH fitting)
    df["garch_var"] = np.nan

    df = df.dropna(subset=["log_ret"])

    # ── Diagnostics ───────────────────────────────────────────────
    ret = df["log_ret"].dropna()
    adf_stat, adf_p, *_ = adfuller(ret, autolag="AIC")
    logger.info(f"  ADF test on log returns: stat={adf_stat:.4f} p={adf_p:.6f} "
                f"{'✅ stationary' if adf_p < 0.05 else '⚠️ non-stationary'}")

    arch_stat, arch_p, *_ = het_arch(ret)
    logger.info(f"  ARCH-LM test:            stat={arch_stat:.4f} p={arch_p:.6f} "
                f"{'✅ ARCH effects → GARCH justified' if arch_p < 0.05 else '⚠️ no ARCH effects'}")

    logger.info(f"  Feature matrix: {df.shape}")
    return df

# ═══════════════════════════════════════════════════════════════════
# CELL 6 — FINANCIAL STRESS INDEX (FSI)
# ═══════════════════════════════════════════════════════════════════

def ffill_fred(fred_df: pd.DataFrame, trade_idx: pd.DatetimeIndex) -> pd.DataFrame:
    if fred_df.empty:
        return pd.DataFrame(index=trade_idx)
    out = pd.DataFrame(index=trade_idx)
    for col in fred_df.columns:
        s = fred_df[col].dropna()
        if len(s) >= 5:
            out[col] = s.reindex(trade_idx, method="ffill")
    return out


def _fsi_validation(fsi: pd.Series, df: pd.DataFrame, tag: str = "") -> dict:
    """
    Three complementary FSI-validity metrics (M2 §4.3 target r>0.60):
      1. Continuous Pearson vs St. Louis Fed Stress Index (STLFSI) — HEADLINE metric.
      2. Point-biserial vs binary NBER recession flag (capped by binary/continuous mismatch).
      3. ROC-AUC of FSI as an NBER-recession-day classifier (discrimination power).
    """
    from sklearn.metrics import roc_auc_score
    f = fsi.fillna(0)
    nber = df["_nber"] if "_nber" in df.columns else pd.Series(0.0, index=df.index)
    out = {}
    r_pb, p_pb = stats.pearsonr(f, nber)
    out["nber_pointbiserial_r"] = round(r_pb, 4)
    try:
        out["nber_roc_auc"] = round(roc_auc_score(nber, f), 4)
    except Exception:
        out["nber_roc_auc"] = np.nan
    if "stl_fsi" in df.columns and df["stl_fsi"].notna().sum() > 100:
        idx = df["stl_fsi"].dropna().index
        r_stl, _ = stats.pearsonr(f.reindex(idx).fillna(0),
                                  df["stl_fsi"].reindex(idx).fillna(0))
        out["stlfsi_r"] = round(r_stl, 4)
        headline = r_stl
        msg = f"STLFSI r={r_stl:.3f}"
    else:
        out["stlfsi_r"] = None
        headline = out["nber_roc_auc"]      # fall back to AUC as the discrimination metric
        msg = f"ROC-AUC={out['nber_roc_auc']:.3f} (STLFSI unavailable)"
    ok = (out.get("stlfsi_r") or 0) >= FSI_CORR_TARGET or (out["nber_roc_auc"] or 0) >= 0.80
    logger.info(f"  FSI validity{tag}: {msg} | NBER r={r_pb:.3f} | "
                f"AUC={out['nber_roc_auc']:.3f} "
                f"{'✅' if ok else '⚠️'}")
    return out


def build_fsi(feat: pd.DataFrame,
              fred_daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    FSI = 0.30×VIX + 0.30×GARCH + 0.20×Drawdown + 0.20×Credit
    GARCH component is zero initially; updated by update_fsi_garch().
    Uses the real FRED high-yield credit spread when available; otherwise a
    VIX-momentum proxy. Validates against STLFSI / NBER (target r > 0.60).
    """
    logger.info("[FSI] Building Financial Stress Index …")
    df = feat.copy()
    if not fred_daily.empty:
        df = df.join(fred_daily, how="left")
        for c in fred_daily.columns:
            df[c] = df[c].ffill()

    sc = MinMaxScaler()
    def norm(s: pd.Series) -> np.ndarray:
        v = s.fillna(s.median()).values.reshape(-1, 1)
        return sc.fit_transform(v).ravel()

    comps: Dict[str, np.ndarray] = {}
    comps["vix"]      = norm(df["vix"])
    comps["garch"]    = np.zeros(len(df))           # placeholder
    comps["drawdown"] = norm(df["drawdown_63"].abs())

    if "credit_spread" in df.columns and df["credit_spread"].notna().sum() > 100:
        comps["credit"] = norm(df["credit_spread"])
        logger.info("  Credit component: real FRED high-yield OAS")
    else:
        vix_z  = ((df["vix"] - df["vix"].rolling(252, min_periods=63).mean()) /
                  df["vix"].rolling(252, min_periods=63).std())
        comps["credit"] = norm(vix_z.clip(lower=0).fillna(0))
        logger.info("  Credit component: VIX-momentum proxy (no FRED)")

    df["FSI"] = (FSI_W["vix"]      * comps["vix"] +
                 FSI_W["garch"]    * comps["garch"] +
                 FSI_W["drawdown"] * comps["drawdown"] +
                 FSI_W["credit"]   * comps["credit"])

    for k, v in comps.items():
        df[f"_fsi_{k}"] = v

    # NBER flag (continuous-vs-binary validation target)
    nber_flag = pd.Series(0, index=df.index, dtype=float)
    for s, e in NBER:
        nber_flag[(df.index >= s) & (df.index <= e)] = 1
    df["_nber"] = nber_flag.values

    _fsi_validation(df["FSI"], df, tag=" (pre-GARCH)")
    logger.info(f"  FSI range: [{df['FSI'].min():.4f}, {df['FSI'].max():.4f}]")
    return df, comps


def update_fsi_garch(df: pd.DataFrame,
                     comps: Dict,
                     garch_var: pd.Series) -> pd.DataFrame:
    """Replace zero GARCH component in FSI with fitted conditional variance."""
    sc = MinMaxScaler()
    df["garch_var"] = garch_var.reindex(df.index).ffill().fillna(0)
    gn = sc.fit_transform(df["garch_var"].values.reshape(-1, 1)).ravel()
    comps["garch"]    = gn
    df["_fsi_garch"]  = gn
    df["FSI"] = (FSI_W["vix"]      * comps["vix"] +
                 FSI_W["garch"]    * gn +
                 FSI_W["drawdown"] * comps["drawdown"] +
                 FSI_W["credit"]   * comps["credit"])
    df.attrs["fsi_validity"] = _fsi_validation(df["FSI"], df, tag=" (final)")
    logger.info(f"  FSI (with GARCH) range: [{df['FSI'].min():.4f}, {df['FSI'].max():.4f}]")
    return df

# ═══════════════════════════════════════════════════════════════════
# CELL 7 — ARMA-GARCH VOLATILITY MODELLING
# ═══════════════════════════════════════════════════════════════════

def _fit_one_garch(returns: pd.Series, spec: dict) -> dict:
    """Fit a single GARCH-family spec and return diagnostics dict."""
    r100 = (returns * 100).dropna()
    vol, p, o, q = spec["vol"], spec["p"], spec["o"], spec["q"]
    label = spec["label"]
    try:
        am  = arch_model(r100, mean="ARX", lags=1,
                         vol=vol, p=p, o=o, q=q,
                         dist="t", rescale=False)
        res = am.fit(disp="off", options={"maxiter": 2000, "ftol": 1e-9})

        # Conditional volatility → back to returns scale
        cond_vol = res.conditional_volatility / 100      # pandas Series
        cond_var = (cond_vol ** 2).rename("garch_var")

        std_r = res.std_resid.dropna()
        lb_p  = sm.stats.diagnostic.acorr_ljungbox(
                    std_r, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
        _, arch_p, *_ = het_arch(std_r)

        logger.info(f"  {label}: BIC={res.bic:.2f} AIC={res.aic:.2f} "
                    f"LB-p={lb_p:.3f} ARCH-p={arch_p:.3f}")

        return dict(label=label, bic=res.bic, aic=res.aic,
                    cond_vol=cond_vol, cond_var=cond_var,
                    lb_p=lb_p, arch_p=arch_p,
                    converged=True, result=res)

    except Exception as e:
        logger.warning(f"  {label} failed: {e}")
        roll = returns.rolling(21).std().fillna(returns.std())
        return dict(label="rolling_std", bic=np.inf, aic=np.inf,
                    cond_vol=roll, cond_var=roll**2,
                    lb_p=np.nan, arch_p=np.nan,
                    converged=False, result=None)


def select_garch(returns: pd.Series) -> Tuple[dict, List[dict]]:
    """
    Test GARCH, GJR-GARCH, EGARCH — select by BIC.
    Huang & Luo (2024): standard GARCH(1,1) wins full-sample;
    asymmetric variants improve crisis sub-samples.
    """
    logger.info("[GARCH] Testing volatility specifications …")
    results = [_fit_one_garch(returns, s) for s in GARCH_SPECS]
    valid   = [r for r in results if r["converged"]]
    best    = min(valid, key=lambda x: x["bic"]) if valid else results[0]
    logger.info(f"  ✅ Selected: {best['label']} (BIC={best['bic']:.2f})")
    return best, results

# ═══════════════════════════════════════════════════════════════════
# CELL 8 — HIDDEN MARKOV MODEL — REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════

def _hmm_bic(model: GaussianHMM, X: np.ndarray) -> float:
    """
    Correct BIC for GaussianHMM. hmmlearn's model.score(X) returns the TOTAL
    log-likelihood (not per-sample), so BIC = -2·logL + k·log(n).
    Free params (full covariance): transitions k(k-1) + means k·d +
    covariances k·d(d+1)/2 + initial (k-1).
    """
    n, d = X.shape
    k    = model.n_components
    np_  = k * (k - 1) + k * d + k * d * (d + 1) // 2 + (k - 1)
    return -2 * model.score(X) + np_ * np.log(n)        # FIXED: no spurious × n


def _fit_hmm_multi(X: np.ndarray, n: int) -> Tuple[GaussianHMM, float, float]:
    """Best GaussianHMM from HMM_N_INIT random seeds (max log-likelihood)."""
    best_m, best_ll = None, -np.inf
    for seed in range(HMM_N_INIT):
        try:
            m = GaussianHMM(n_components=n, covariance_type="full",
                            n_iter=HMM_N_ITER, tol=1e-5,
                            random_state=seed)
            m.fit(X)
            ll = m.score(X)
            if ll > best_ll:
                best_ll, best_m = ll, m
        except Exception:
            pass
    if best_m is None:
        raise RuntimeError(f"All HMM seeds failed for n={n}")
    bic = _hmm_bic(best_m, X)
    logger.info(f"  HMM n={n}: LL={best_ll:.2f}  BIC={bic:.2f}")
    return best_m, best_ll, bic


# HMM features — must match col names from engineer_features()
HMM_FEATURES = ["log_ret", "garch_var", "vol_21d", "FSI"]


def select_hmm(feat: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Fit n∈{2,3,4} (BIC profile reported in figure 05) but RETAIN the canonical
    HMM_FORCE_N=3 model for all downstream regime labelling. See HMM_FORCE_N note:
    with N≈8,700 obs, BIC drops monotonically with n, so a fixed economically
    interpretable 3-state model (stable/volatile/crisis) is the principled choice
    (Ang & Timmermann 2012).
    """
    logger.info("[HMM] Testing regime models (BIC profile n=2,3,4) …")
    fcols  = [c for c in HMM_FEATURES if c in feat.columns]
    Xdf    = feat[fcols].dropna()
    scaler = StandardScaler()
    X      = scaler.fit_transform(Xdf)
    dates  = Xdf.index

    all_res = {}
    for n in HMM_N_LIST:
        try:
            m, ll, bic = _fit_hmm_multi(X, n)
            all_res[n] = dict(model=m, ll=ll, bic=bic,
                              scaler=scaler, X=X, dates=dates, fcols=fcols)
        except Exception as e:
            logger.warning(f"  n={n} failed: {e}")

    bic_min = min(all_res, key=lambda k: all_res[k]["bic"])
    chosen  = HMM_FORCE_N if HMM_FORCE_N in all_res else bic_min
    logger.info(f"  BIC-min n={bic_min}; RETAINED canonical n={chosen} "
                f"(interpretable stable/volatile/crisis)")

    with open(MODEL_DIR / "hmm_best.pkl", "wb") as f:
        pickle.dump(all_res[chosen], f)

    return all_res[chosen], all_res


def label_states(model: GaussianHMM, X: np.ndarray,
                 fcols: List[str]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Rank states by mean volatility (vol_21d or first feature).
    Lowest  → 0 Stable | Middle → 1 Volatile | Highest → 2 Crisis
    """
    k = model.n_components
    d = min(len(fcols), X.shape[1])
    means = pd.DataFrame(model.means_[:, :d], columns=fcols[:d])
    vc    = "vol_21d" if "vol_21d" in means.columns else means.columns[0]
    order = means[vc].argsort().values          # ascending volatility
    state_map = {order[i]: i for i in range(k)}

    raw    = model.predict(X)
    labels = np.vectorize(state_map.get)(raw)

    probs_raw = model.predict_proba(X)
    probs     = np.zeros_like(probs_raw)
    for rs, ss in state_map.items():
        if ss < probs.shape[1]:
            probs[:, ss] = probs_raw[:, rs]

    return labels, probs, state_map


def build_regime_df(best: dict) -> pd.DataFrame:
    model, X, dates = best["model"], best["X"], best["dates"]
    fcols           = best["fcols"]
    labels, probs, state_map = label_states(model, X, fcols)
    k = model.n_components

    d: dict = {"regime": labels}
    name_map = {0: "prob_stable", 1: "prob_volatile", 2: "prob_crisis"}
    for i in range(k):
        col = name_map.get(i, f"prob_s{i}")
        d[col] = probs[:, i] if i < probs.shape[1] else 0.0

    rdf = pd.DataFrame(d, index=dates)
    for s, nm in [(0,"Stable"),(1,"Volatile"),(2,"Crisis")]:
        pct = (rdf["regime"] == s).mean() * 100
        logger.info(f"  {nm}: {pct:.1f}%")
    return rdf

# ═══════════════════════════════════════════════════════════════════
# CELL 9 — FINBERT SENTIMENT PIPELINE
# ═══════════════════════════════════════════════════════════════════

def load_finbert():
    """Load ProsusAI/finbert to DEVICE. FP16 on GPU for ~2× throughput."""
    logger.info(f"[NLP] Loading FinBERT on {DEVICE} …")
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    mdl = mdl.to(DEVICE).eval()
    if DEVICE.type == "cuda":
        mdl = mdl.half()
        logger.info("  FP16 mode enabled")
    # Label order for ProsusAI/finbert: [positive(0), negative(1), neutral(2)]
    logger.info("  FinBERT ready ✅")
    return tok, mdl


@torch.no_grad()
def _fb_batch(texts: List[str], tok, mdl) -> np.ndarray:
    """Single-batch inference → (n,3) float32 softmax probs."""
    enc = tok(texts, padding=True, truncation=True,
               max_length=FINBERT_MAXLEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    return F.softmax(mdl(**enc).logits.float(), dim=-1).cpu().numpy()


def run_finbert(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full GPU-accelerated FinBERT inference with caching + checkpointing.
    ProsusAI/finbert output: index-0=positive, index-1=negative, index-2=neutral.
    """
    p = _cp("finbert_scores")
    if p.exists():
        logger.info("[NLP] FinBERT scores from cache ✅")
        df = pd.read_csv(p, parse_dates=["date"])
        return df

    if news_df.empty:
        logger.warning("[NLP] No news → returning empty sentiment")
        return pd.DataFrame(columns=["date","headline","p_pos","p_neg","p_neu"])

    tok, mdl = load_finbert()
    texts    = news_df["headline"].tolist()
    CKPT     = CACHE_DIR / "fb_ckpt.npy"

    all_probs = []
    start_i   = 0
    if CKPT.exists():
        prev = np.load(CKPT)
        all_probs.append(prev)
        start_i = len(prev)
        logger.info(f"  Resuming from checkpoint idx {start_i}")

    for i in tqdm(range(start_i, len(texts), FINBERT_BATCH),
                  desc="FinBERT", unit="batch"):
        batch = texts[i: i + FINBERT_BATCH]
        try:
            p_ = _fb_batch(batch, tok, mdl)
        except Exception:
            p_ = np.full((len(batch), 3), 1/3, dtype=np.float32)
        all_probs.append(p_)
        # Checkpoint every 5 000 headlines
        if (i + FINBERT_BATCH) % 5000 == 0:
            np.save(CKPT, np.vstack(all_probs))

    arr = np.vstack(all_probs)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = news_df[["date","headline"]].copy()
    out["p_pos"] = arr[:, 0]   # positive class (FinBERT index 0)
    out["p_neg"] = arr[:, 1]   # negative class (FinBERT index 1)
    out["p_neu"] = arr[:, 2]   # neutral  class (FinBERT index 2)
    out.to_csv(p, index=False)
    logger.info(f"  Saved {len(out):,} FinBERT scores ✅")
    return out


def aggregate_sentiment(scores: pd.DataFrame,
                        trade_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Per-headline scores → daily fear_index, panic_signal, rolling windows.
    Also computes Tetlock (2007) sentiment composite = mean(pos) - mean(neg).
    """
    if scores.empty:
        df = pd.DataFrame(0.0, index=trade_idx,
                          columns=["fear_index","panic_signal","headline_count",
                                   "sentiment_comp","fear_3d","fear_7d","fear_21d"])
        return df

    sc = scores.copy()
    sc["date"] = pd.to_datetime(sc["date"]).dt.tz_localize(None).dt.normalize()

    daily = (
        sc.groupby("date")
        .agg(
            fear_index     = ("p_neg", "mean"),
            p_neg_max      = ("p_neg", "max"),
            p_neg_med      = ("p_neg", "median"),
            pos_mean       = ("p_pos", "mean"),
            headline_count = ("headline", "count"),
        )
        .reset_index()
    )
    daily["sentiment_comp"] = daily["pos_mean"] - daily["fear_index"]
    daily["panic_signal"]   = (daily["fear_index"] > PANIC_THR).astype(int)
    daily = daily.set_index("date")

    # Align to trading calendar (forward-fill weekends/holidays)
    daily = daily.reindex(trade_idx, method="ffill")
    daily["fear_index"]     = daily["fear_index"].fillna(daily["fear_index"].median())
    daily["panic_signal"]   = daily["panic_signal"].fillna(0).astype(int)
    daily["headline_count"] = daily["headline_count"].fillna(0)

    daily["fear_3d"]  = daily["fear_index"].rolling(3,  min_periods=1).mean()
    daily["fear_7d"]  = daily["fear_index"].rolling(7,  min_periods=1).mean()
    daily["fear_21d"] = daily["fear_index"].rolling(21, min_periods=1).mean()

    cov = (daily["headline_count"] > 0).mean()
    logger.info(f"  News trading-day coverage: {cov:.1%}")
    logger.info(f"  Fear index: [{daily['fear_index'].min():.4f}, {daily['fear_index'].max():.4f}]")
    return daily


def build_synthetic_sentiment(feat: pd.DataFrame) -> pd.DataFrame:
    """
    VIX z-score + negative return shock → synthetic fear proxy [0,1].
    Used to fill gaps when news data is unavailable (e.g. 2020/2022 if
    primary Kaggle dataset only covers 2008-2016).
    Clearly flagged with is_synthetic=1 in all outputs.
    """
    vix  = feat["vix"]
    ret  = feat["log_ret"]

    vix_z = ((vix - vix.rolling(252, min_periods=63).mean()) /
              vix.rolling(252, min_periods=63).std()).clip(-3, 5)
    vix_s = (vix_z - vix_z.min()) / (vix_z.max() - vix_z.min() + 1e-9)

    ret_z = ((ret - ret.rolling(63, min_periods=21).mean()) /
              ret.rolling(63, min_periods=21).std())
    neg_s = (-ret_z).clip(lower=0)
    neg_s /= (neg_s.max() + 1e-9)

    df = pd.DataFrame(index=feat.index)
    df["fear_index"]     = (0.70 * vix_s + 0.30 * neg_s).clip(0, 1).fillna(0)
    df["panic_signal"]   = (df["fear_index"] > PANIC_THR).astype(int)
    df["fear_3d"]        = df["fear_index"].rolling(3).mean()
    df["fear_7d"]        = df["fear_index"].rolling(7).mean()
    df["fear_21d"]       = df["fear_index"].rolling(21).mean()
    df["headline_count"] = 0
    df["sentiment_comp"] = 1 - df["fear_index"]
    df["is_synthetic"]   = 1
    return df.fillna(0)


def run_vader(news_df: pd.DataFrame,
              trade_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    VADER lexicon baseline — per Shobayo et al. (2024).
    Compare correlation with FSI vs FinBERT.
    """
    if news_df.empty:
        return pd.DataFrame(index=trade_idx,
                            columns=["vader_fear","vader_comp"])
    logger.info("[NLP] Running VADER baseline …")
    va = SentimentIntensityAnalyzer()
    sc = news_df.copy()
    sc["vader_neg"]  = sc["headline"].apply(
        lambda x: va.polarity_scores(str(x))["neg"])
    sc["vader_comp"] = sc["headline"].apply(
        lambda x: va.polarity_scores(str(x))["compound"])
    sc["date"] = pd.to_datetime(sc["date"]).dt.tz_localize(None).dt.normalize()
    daily = (sc.groupby("date")
               .agg(vader_fear=("vader_neg","mean"),
                    vader_comp=("vader_comp","mean"))
               .reindex(trade_idx, method="ffill")
               .fillna(0))
    logger.info("  VADER done ✅")
    return daily

# ═══════════════════════════════════════════════════════════════════
# CELL 10 — LEAD-LAG CROSS-CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def cross_corr(x: pd.Series, y: pd.Series,
               max_lag: int = MAX_LAG,
               n_boot: int = BOOT_N) -> dict:
    """
    Pearson cross-correlation at lags from -max_lag to +max_lag.
    Positive lag k: y leads x by k days.
    Bootstrap 95 % CI on the peak-lag estimate.
    """
    idx = x.index.intersection(y.index)
    xv  = x.reindex(idx).ffill().fillna(0).values.astype(float)
    yv  = y.reindex(idx).ffill().fillna(0).values.astype(float)
    n   = len(idx)
    lags = np.arange(-max_lag, max_lag + 1)

    corrs = []
    for lag in lags:
        if lag >= 0:
            r = np.corrcoef(xv[lag:], yv[:n-lag])[0, 1] if n > lag else np.nan
        else:
            r = np.corrcoef(xv[:n+lag], yv[-lag:])[0, 1] if n > -lag else np.nan
        corrs.append(r if np.isfinite(r) else 0.0)
    corrs = np.array(corrs)

    pi       = int(np.argmax(np.abs(corrs)))
    peak_lag = int(lags[pi])
    peak_r   = float(corrs[pi])

    # Bootstrap on peak lag
    boot_lags = []
    for _ in range(n_boot):
        idx_ = np.random.choice(n, n, replace=True)
        xb, yb = xv[idx_], yv[idx_]
        bc = []
        for lag in lags:
            if lag >= 0 and n > lag:
                bc.append(np.corrcoef(xb[lag:], yb[:n-lag])[0,1])
            elif lag < 0 and n > -lag:
                bc.append(np.corrcoef(xb[:n+lag], yb[-lag:])[0,1])
            else:
                bc.append(0.0)
        bc = np.array(bc)
        bc = np.where(np.isfinite(bc), bc, 0.0)
        boot_lags.append(int(lags[np.argmax(np.abs(bc))]))

    ci_lo = float(np.percentile(boot_lags, 2.5))
    ci_hi = float(np.percentile(boot_lags, 97.5))

    if peak_lag > 0:
        interp = f"Sentiment LEADS price-regime by {peak_lag} trading days"
    elif peak_lag < 0:
        interp = f"Price-regime LEADS sentiment by {abs(peak_lag)} trading days"
    else:
        interp = "Contemporaneous (peak lag = 0)"

    return dict(lags=lags, corrs=corrs, peak_lag=peak_lag,
                peak_r=peak_r, ci_lo=ci_lo, ci_hi=ci_hi, interp=interp)


def run_all_lead_lag(feat: pd.DataFrame, sent: pd.DataFrame) -> dict:
    fsi  = feat["FSI"]
    fear = sent["fear_index"]
    res  = {"overall": cross_corr(fsi, fear)}
    logger.info(f"  Overall: {res['overall']['interp']} "
                f"(r={res['overall']['peak_r']:.4f})")
    for name, (s, e) in CRISIS_WINDOWS.items():
        pre  = pd.Timestamp(s) - pd.DateOffset(months=6)
        fw   = fsi[(fsi.index >= pre)  & (fsi.index <= e)]
        fw2  = fear[(fear.index >= pre) & (fear.index <= e)]
        if len(fw) < 60 or len(fw2) < 20:
            continue
        r = cross_corr(fw, fw2, n_boot=200)
        res[name] = r
        logger.info(f"  {name}: {r['interp']} (r={r['peak_r']:.4f})")
    return res


def run_granger(fsi: pd.Series, fear: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """
    Granger causality: does fear Granger-cause FSI?
    Replicates Bollen et al. (2011) framework.
    """
    idx  = fsi.index.intersection(fear.index)
    data = pd.DataFrame({"fsi": fsi.loc[idx], "fear": fear.loc[idx]}).dropna()
    rows = []
    try:
        gc = grangercausalitytests(data[["fsi","fear"]], maxlag=max_lag, verbose=False)
        for lag, res in gc.items():
            f, p = res[0]["params_ftest"][:2]
            rows.append({"lag": lag, "f_stat": round(f,4),
                         "p_value": round(p,4), "sig": p < 0.05})
    except Exception as e:
        logger.warning(f"  Granger failed: {e}")
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════
# CELL 11 — MULTIMODAL FUSION MODEL
# ═══════════════════════════════════════════════════════════════════

FUSION_FEATURE_COLS = [
    "prob_stable", "prob_volatile", "prob_crisis",
    "fear_index", "fear_3d", "fear_7d", "panic_signal",
    "FSI", "vol_21d", "vix", "drawdown_63",
]


def build_fusion(regime: pd.DataFrame,
                 sent: pd.DataFrame,
                 feat: pd.DataFrame) -> pd.DataFrame:
    """
    Combine HMM posteriors + sentiment + price features.
    Target: does HMM enter Crisis state within PRED_HORIZON trading days?
    Uses shift(-PRED_HORIZON) to avoid look-ahead bias.
    """
    idx = (regime.index.intersection(sent.index).intersection(feat.index))
    f   = pd.DataFrame(index=idx)

    for col in ["prob_stable","prob_volatile","prob_crisis"]:
        if col in regime.columns:
            f[col] = regime[col].reindex(idx)

    for col in ["fear_index","fear_3d","fear_7d","panic_signal"]:
        if col in sent.columns:
            f[col] = sent[col].reindex(idx).fillna(0)

    for col in ["FSI","vol_21d","vix","drawdown_63"]:
        if col in feat.columns:
            f[col] = feat[col].reindex(idx)

    crisis_now = (regime["regime"] == 2).astype(int)
    f["target"] = crisis_now.reindex(idx).shift(-PRED_HORIZON).fillna(0).astype(int)
    f = f.dropna()

    pos = f["target"].mean()
    logger.info(f"  Fusion matrix: {f.shape}  crisis-class rate: {pos:.2%}")
    return f


def _tune_threshold(model, X, y) -> float:
    """Pick the probability threshold that maximises F1 on the TRAINING data
    only (no crisis-window leakage). Falls back to 0.5 if degenerate."""
    try:
        pr = model.predict_proba(X)[:, 1]
        p, r, t = precision_recall_curve(y, pr)
        f1 = 2 * p * r / (p + r + 1e-9)
        if len(t) == 0:
            return 0.5
        return float(t[int(np.argmax(f1[:-1]))])
    except Exception:
        return 0.5


def train_models(fusion: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Event-based holdout: train ONLY on non-crisis periods; evaluate on each
    crisis window. Logistic Regression is wrapped in a StandardScaler pipeline
    (features span very different scales: VIX~10-80 vs probs 0-1). Each model's
    decision threshold is tuned on the training set to maximise F1, then applied
    unchanged to the held-out crisis windows. Three models compared.
    """
    fcols = [c for c in fusion.columns if c != "target"]
    X, y  = fusion[fcols].values, fusion["target"].values
    dates = fusion.index

    train_mask = np.ones(len(fusion), dtype=bool)
    for s, e in CRISIS_WINDOWS.values():
        train_mask &= ~((dates >= s) & (dates <= e))
    Xtr, ytr = X[train_mask], y[train_mask]
    logger.info(f"  Training: {Xtr.shape[0]} non-crisis samples  "
                f"(crisis={ytr.mean():.2%})  features={len(fcols)}")

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                               class_weight="balanced", max_iter=2000,
                               random_state=SEED)),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED),
    }
    thresholds: Dict[str, float] = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        thresholds[name] = _tune_threshold(m, Xtr, ytr)
        with open(MODEL_DIR / f"fusion_{name.replace(' ','_').lower()}.pkl", "wb") as f_:
            pickle.dump({"model": m, "threshold": thresholds[name],
                         "features": fcols}, f_)
        logger.info(f"  {name}: tuned threshold = {thresholds[name]:.3f}")

    eval_out: dict = {}
    for crisis, (s, e) in CRISIS_WINDOWS.items():
        mask = (dates >= s) & (dates <= e)
        Xe, ye = X[mask], y[mask]
        if len(Xe) == 0 or ye.sum() == 0:
            logger.warning(f"  {crisis}: no positive labels in window — skipped")
            continue
        cr: dict = {}
        for name, m in models.items():
            yprob = m.predict_proba(Xe)[:, 1]
            yp    = (yprob >= thresholds[name]).astype(int)
            f1    = f1_score(ye, yp, zero_division=0)
            prec  = precision_score(ye, yp, zero_division=0)
            rec   = recall_score(ye, yp, zero_division=0)
            try:    auc = roc_auc_score(ye, yprob)
            except Exception: auc = np.nan
            cr[name] = dict(f1=round(f1,4), prec=round(prec,4),
                            rec=round(rec,4), auc=round(auc,4))
            ok = "✅" if f1 >= FUSION_F1_TARGET else "⚠️ "
            logger.info(f"  {ok} {crisis} | {name}: "
                        f"F1={f1:.4f}  Prec={prec:.4f}  "
                        f"Rec={rec:.4f}  AUC={auc:.4f}")
        eval_out[crisis] = cr
    return {"models": models, "thresholds": thresholds, "features": fcols}, eval_out

# ═══════════════════════════════════════════════════════════════════
# CELL 12 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════

def run_shap(fusion: pd.DataFrame, trained: dict) -> Tuple[dict, pd.DataFrame]:
    """
    Per-crisis SHAP attribution using the Random Forest TreeExplainer (exact for
    tree ensembles, scale-invariant, and robust). Identifies which signal — price
    vs sentiment — drives each crisis transition (Bussmann 2020; Lundberg 2020).
    """
    logger.info("[SHAP] Computing feature attributions …")
    fcols = trained.get("features", [c for c in fusion.columns if c != "target"])
    X     = pd.DataFrame(fusion[fcols].values, columns=fcols, index=fusion.index)
    models = trained["models"]
    out: dict = {"by_crisis": {}}

    rf = models["Random Forest"]
    try:
        rf_e = shap.TreeExplainer(rf)
        rf_v = rf_e.shap_values(X)
        if isinstance(rf_v, list):              # [class0, class1] → take crisis class
            rf_v = rf_v[1]
        elif rf_v.ndim == 3:                    # (n, features, classes)
            rf_v = rf_v[:, :, 1]
        out["rf"] = {"values": rf_v, "cols": fcols}
    except Exception as e:
        logger.warning(f"  TreeExplainer global failed: {e}")
        rf_e = None

    for crisis, (s, e) in CRISIS_WINDOWS.items():
        mask = (X.index >= s) & (X.index <= e)
        if mask.sum() == 0 or rf_e is None:
            continue
        try:
            v = rf_e.shap_values(X[mask])
            if isinstance(v, list):
                v = v[1]
            elif v.ndim == 3:
                v = v[:, :, 1]
            ma = pd.Series(np.abs(v).mean(axis=0),
                           index=fcols).sort_values(ascending=False)
            out["by_crisis"][crisis] = ma
            logger.info(f"  {crisis} top-3: "
                        f"{ {k: round(val,4) for k,val in ma.head(3).items()} }")
        except Exception as ex:
            logger.warning(f"  SHAP {crisis} failed: {ex}")
    return out, X

# ═══════════════════════════════════════════════════════════════════
# CELL 13 — RESEARCH PAPER COMPARISON BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

def _trading_lead(regime: pd.DataFrame, onset: pd.Timestamp,
                  lookback_td: int = 60, lookahead_td: int = 10) -> Tuple:
    """
    First crisis-state (==2) day within [onset - lookback_td, onset + lookahead_td]
    trading days. Returns (first_date, lead_trading_days, detected_by_deadline).
    Positive lead = detected BEFORE onset (early warning = good).
    """
    idx = regime.index
    pos = idx.searchsorted(onset)
    lo  = max(0, pos - lookback_td)
    hi  = min(len(idx) - 1, pos + lookahead_td)
    win = regime.iloc[lo:hi + 1]
    cdays = win.index[win["regime"] == 2]
    if len(cdays) == 0:
        return None, None, False
    first = cdays[0]
    lead_td = int(pos - idx.searchsorted(first))      # >0 → before onset
    return first, lead_td, True


def benchmark_wang2025(regime: pd.DataFrame) -> pd.DataFrame:
    """
    Wang et al. (2025) baseline: HMM-only early-warning lead time (trading days).
    Detection counts as a PASS if the crisis state appears no later than 10
    trading days after the official onset; earlier detection is better.
    """
    rows = []
    for crisis, (s, e) in CRISIS_WINDOWS.items():
        first, lead, ok = _trading_lead(regime, pd.Timestamp(s))
        rows.append({
            "Crisis": crisis,
            "Detected": "✅" if ok else "❌",
            "First": str(first.date()) if first is not None else "N/A",
            "Crisis_start": s,
            "Lead_td": lead,                       # +ve = early warning
            "By onset+10td": "✅" if ok else "❌",
        })
    df = pd.DataFrame(rows)
    logger.info("[BENCH] Wang2025 (HMM-only early warning):\n" + df.to_string(index=False))
    return df


def compare_finbert_vader(sent_fb: pd.DataFrame,
                           sent_vader: pd.DataFrame,
                           fsi: pd.Series) -> dict:
    """
    FinBERT vs VADER correlation with FSI.
    Shobayo et al. (2024): FinBERT outperforms VADER on crisis text.
    """
    if sent_vader.empty or "vader_fear" not in sent_vader.columns:
        return {}
    idx = (fsi.index.intersection(sent_fb.index)
               .intersection(sent_vader.index))
    f0  = fsi.reindex(idx).fillna(0)
    fb  = sent_fb["fear_index"].reindex(idx).fillna(0)
    vd  = sent_vader["vader_fear"].reindex(idx).fillna(0)
    r_fb, p_fb = stats.pearsonr(f0, fb)
    r_vd, p_vd = stats.pearsonr(f0, vd)
    winner = "FinBERT" if abs(r_fb) > abs(r_vd) else "VADER"
    res = dict(finbert_r=round(r_fb,4), finbert_p=round(p_fb,4),
               vader_r=round(r_vd,4),   vader_p=round(p_vd,4),
               winner=winner,
               interp=f"{winner} wins | FinBERT r={r_fb:.4f}  VADER r={r_vd:.4f}")
    logger.info(f"[BENCH] {res['interp']}")
    return res


def validate_checklist(regime: pd.DataFrame,
                        sent: pd.DataFrame,
                        eval_res: dict) -> pd.DataFrame:
    """
    M2 Section 4.5 validation:
      Req 1: HMM enters State 2 within ±10 trading days of crisis onset.
      Req 2: Panic signal fires before HMM State 2.
      Req 3: Best F1 ≥ 0.70 on held-out crisis window.
    """
    rows = []
    for crisis, (s, e) in CRISIS_WINDOWS.items():
        start = pd.Timestamp(s)

        # Req 1 — HMM crisis state detected by onset + 10 trading days
        first, lead_td, req1 = _trading_lead(regime, start)

        # Req 2 — panic signal fires in the 30 calendar days before onset
        pre  = sent[(sent.index >= start - pd.Timedelta(days=30)) &
                    (sent.index < start)]
        p_before = bool((pre["panic_signal"] == 1).any()) \
                   if "panic_signal" in sent.columns else None

        # Req 3 — best held-out F1 ≥ 0.70
        if crisis in eval_res:
            f1s = [m["f1"] for m in eval_res[crisis].values()]
            best_f1 = max(f1s) if f1s else None
        else:
            best_f1 = None
        req3 = bool(best_f1 is not None and best_f1 >= FUSION_F1_TARGET)

        rows.append({
            "Crisis":        crisis,
            "Period":        f"{s} → {e}",
            "HMM ≤onset+10td": "✅" if req1 else "❌",
            "First detect":  str(first.date()) if first is not None else "—",
            "Lead (td)":     lead_td,
            "Panic before":  ("✅" if p_before else "❌") if p_before is not None else "—",
            "Best F1":       f"{best_f1:.4f}" if best_f1 else "—",
            "F1 ≥ 0.70":     "✅" if req3 else "❌",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("  CRISIS VALIDATION CHECKLIST (M2 Section 4.5)")
    print("=" * 72)
    print(df.to_string(index=False))
    return df

# ═══════════════════════════════════════════════════════════════════
# CELL 14 — INDIVIDUAL STOCK ANALYSIS  (AAPL / JPM / XOM)
# ═══════════════════════════════════════════════════════════════════

def analyse_stocks(market: dict, feat: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sector generalisation check: refit a fresh HMM on each stock
    (not reuse index-fitted model) and check crisis-state coincidence
    with the three validation windows. Per M2 Section 3.1.
    """
    logger.info("[STOCKS] Cross-sector analysis …")
    rows = []
    for ticker in STOCK_TICKERS:
        t = ticker.lower()
        if t not in market or "Close" not in market[t].columns:
            continue
        stk = market[t]
        try:
            df = pd.DataFrame(index=stk.index)
            df["log_ret"]     = np.log(stk["Close"] / stk["Close"].shift(1))
            df["vol_21d"]     = df["log_ret"].rolling(21).std() * np.sqrt(252)
            df["drawdown_63"] = (stk["Close"].rolling(63)
                                 .apply(lambda x: (x[-1]-x.max())/x.max()
                                        if x.max() != 0 else 0, raw=True))
            df["vix"]         = feat["vix"].reindex(df.index).ffill()
            df["FSI"]         = feat["FSI"].reindex(df.index).ffill()
            df["garch_var"]   = feat["garch_var"].reindex(df.index).ffill()
            df = df.dropna()

            fcols = [c for c in HMM_FEATURES if c in df.columns]
            Xdf   = df[fcols]
            sc_   = StandardScaler()
            X_    = sc_.fit_transform(Xdf)

            # Fresh HMM for this stock (3 states, 20 seeds)
            best_m, best_ll = None, -np.inf
            for seed in range(20):
                try:
                    m = GaussianHMM(n_components=3, covariance_type="full",
                                    n_iter=100, random_state=seed)
                    m.fit(X_)
                    ll = m.score(X_)
                    if ll > best_ll:
                        best_ll, best_m = ll, m
                except Exception:
                    pass
            if best_m is None:
                continue

            labels_, _, _ = label_states(best_m, X_, fcols)

            for crisis, (s, e) in CRISIS_WINDOWS.items():
                idx_  = Xdf.index
                win_  = (idx_ >= s) & (idx_ <= e)
                if win_.sum() == 0:
                    continue
                pct = float((labels_[win_] == 2).mean())
                rows.append({"Ticker": ticker, "Crisis": crisis,
                             "Pct_crisis_state": round(pct, 4)})
        except Exception as ex:
            logger.warning(f"  {ticker}: {ex}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print("\n[STOCKS] Cross-sector regime coincidence:")
        print(df_out.pivot(index="Crisis", columns="Ticker",
                           values="Pct_crisis_state").to_string())
    return df_out


def vn_market_robustness() -> Optional[pd.DataFrame]:
    """
    OPTIONAL appendix (out of core S&P 500 scope): apply the 3-state HMM regime
    detector to the attached Vietnamese-market quant DB to test cross-market
    generalisation. Runs ONLY if a compatible OHLCV CSV is found; never crashes.
    """
    inp = Path("/kaggle/input")
    if not inp.exists():
        return None
    cand = [c for c in inp.rglob("*.csv")
            if any(h in str(c).lower() for h in VN_DATASET_HINTS)
            and c.stat().st_size > 10_000]
    if not cand:
        return None
    logger.info(f"[VN] Emerging-market robustness appendix — {len(cand)} candidate file(s)")
    rows = []
    for csv in cand[:3]:
        try:
            df = pd.read_csv(csv, nrows=500_000)
            cols = {c.lower().strip(): c for c in df.columns}
            dcol = next((cols[c] for c in cols if c in
                         {"date", "datetime", "time", "trading_date", "tradingdate"}), None)
            ccol = next((cols[c] for c in cols if c in
                         {"close", "close_price", "adj_close", "closeprice"}), None)
            if not (dcol and ccol):
                continue
            g = pd.DataFrame()
            g["close"] = pd.to_numeric(df[ccol], errors="coerce")
            g.index = pd.to_datetime(df[dcol], errors="coerce")
            g = g[~g.index.isna()].sort_index()
            g = g[~g.index.duplicated(keep="last")].dropna()
            if len(g) < 500:
                continue
            g["log_ret"]  = np.log(g["close"] / g["close"].shift(1))
            g["vol_21d"]  = g["log_ret"].rolling(21).std() * np.sqrt(252)
            g = g.dropna()
            Xv = StandardScaler().fit_transform(g[["log_ret", "vol_21d"]])
            best_m, best_ll = None, -np.inf
            for seed in range(15):
                try:
                    m = GaussianHMM(n_components=3, covariance_type="full",
                                    n_iter=100, random_state=seed)
                    m.fit(Xv); ll = m.score(Xv)
                    if ll > best_ll:
                        best_ll, best_m = ll, m
                except Exception:
                    pass
            if best_m is None:
                continue
            means = pd.DataFrame(best_m.means_, columns=["log_ret", "vol_21d"])
            order = means["vol_21d"].argsort().values
            smap  = {order[i]: i for i in range(3)}
            labs  = np.vectorize(smap.get)(best_m.predict(Xv))
            reg   = pd.Series(labs, index=g.index)
            # COVID coincidence in VN market (2020 window present in 2014-2024 data)
            cov = reg[(reg.index >= "2020-02-01") & (reg.index <= "2020-05-31")]
            share = float((cov == 2).mean()) if len(cov) else np.nan
            rows.append({"File": csv.name, "Rows": len(g),
                         "Span": f"{g.index.min().date()}→{g.index.max().date()}",
                         "COVID_crisis_share": round(share, 4)})
            logger.info(f"  ✓ {csv.name}: COVID crisis-state share={share:.2%}")
        except Exception as ex:
            logger.debug(f"  VN skip {csv.name}: {ex}")
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print("\n[VN] Emerging-market (Vietnam) regime robustness:")
        print(df_out.to_string(index=False))
    return df_out if not df_out.empty else None

# ═══════════════════════════════════════════════════════════════════
# CELL 15 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════

def _shade_crises(ax, alpha=0.10, label=True):
    names = list(CRISIS_WINDOWS.keys())
    for i, (nm, (s, e)) in enumerate(CRISIS_WINDOWS.items()):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color="red", alpha=alpha, zorder=1)
        if label and i == 0:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       color="red", alpha=alpha, zorder=1,
                       label="Crisis window")


def plot_regime_timeline(feat: pd.DataFrame, regime: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    idx   = feat.index.intersection(regime.index)
    price = feat["close"].loc[idx]
    reg   = regime["regime"].loc[idx]
    fsi   = feat["FSI"].loc[idx]

    a1.plot(price.index, price, color="#2C3E50", lw=0.7, zorder=5)
    a1.set_yscale("log")
    a1.set_title("S&P 500 with HMM Regime Labels (1990–2024)",
                 fontsize=14, fontweight="bold")
    a1.set_ylabel("S&P 500 (log scale)")

    sc_col  = {0: C["stable"], 1: C["volatile"], 2: C["crisis"]}
    sc_alp  = {0: 0.12,        1: 0.22,          2: 0.35}
    sc_lbl  = {0: "Stable",    1: "Volatile",     2: "Crisis"}

    for state in [0, 1, 2]:
        m  = (reg == state)
        st = m.index[m & ~m.shift(1, fill_value=False)]
        en = m.index[m & ~m.shift(-1, fill_value=False)]
        for s_, e_ in zip(st, en):
            a1.axvspan(s_, e_, color=sc_col[state],
                       alpha=sc_alp[state], zorder=2)

    for nm, (s, e) in CRISIS_WINDOWS.items():
        a1.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color="red", alpha=0.06, zorder=3)
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        a1.text(mid, price.quantile(0.90), nm.replace("_","\n"),
                ha="center", va="top", fontsize=7, color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    patches = [mpatches.Patch(color=sc_col[s], label=sc_lbl[s], alpha=0.6)
               for s in range(3)]
    a1.legend(handles=patches, loc="upper left")

    a2.fill_between(fsi.index, fsi.values, color=C["fsi"], alpha=0.55)
    a2.set_ylabel("FSI"); a2.set_ylim(0, 1)
    a2.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.6)
    a2.set_xlabel("Date")
    _shade_crises(a2, alpha=0.08, label=False)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_regime_timeline.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  01_regime_timeline.png")


def plot_sentiment_fsi(feat: pd.DataFrame, sent: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    idx = feat.index.intersection(sent.index)

    for ax in axes:
        _shade_crises(ax, alpha=0.09, label=False)

    ax = axes[0]
    vix = feat["vix"].loc[idx]
    ax.plot(vix.index, vix.values, color=C["crisis"], lw=0.8)
    ax.fill_between(vix.index, vix.values, alpha=0.25, color=C["crisis"])
    ax.axhline(30, color="gray", ls="--", lw=0.8, label="VIX = 30")
    ax.set_ylabel("VIX"); ax.legend(fontsize=9)
    ax.set_title("CBOE VIX Fear Gauge", fontweight="bold")

    ax = axes[1]
    fi = sent["fear_index"].reindex(idx)
    ax.plot(fi.index, fi.values, color=C["sentiment"], lw=0.8)
    ax.fill_between(fi.index, fi.values, alpha=0.25, color=C["sentiment"])
    ax.axhline(PANIC_THR, color="orange", ls="--", lw=0.9,
               label=f"Panic threshold ({PANIC_THR})")
    ax.set_ylim(0, 1); ax.set_ylabel("Fear Index"); ax.legend(fontsize=9)
    ax.set_title("FinBERT Sentiment Fear Index", fontweight="bold")

    ax = axes[2]
    fsi = feat["FSI"].reindex(idx)
    ax.plot(fsi.index, fsi.values, color=C["fsi"], lw=0.8)
    ax.fill_between(fsi.index, fsi.values, alpha=0.30, color=C["fsi"])
    ax.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_ylim(0, 1); ax.set_ylabel("FSI [0–1]"); ax.set_xlabel("Date")
    ax.set_title("Financial Stress Index (Composite)", fontweight="bold")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_sentiment_vs_fsi.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  02_sentiment_vs_fsi.png")


def plot_lead_lag(res: dict) -> None:
    keys = list(res.keys())
    n    = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        r = res[key]
        lags, corrs = r["lags"], r["corrs"]
        cols = [C["sentiment"] if l > 0 else C["fsi"] for l in lags]
        ax.bar(lags, corrs, color=cols, alpha=0.7, width=0.85)
        ax.axvline(r["peak_lag"], color="red", ls="--", lw=1.5,
                   label=f"Peak = {r['peak_lag']}d")
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.text(0.05, 0.97,
                f"95% CI: [{r['ci_lo']:.0f}, {r['ci_hi']:.0f}]d\n"
                f"r = {r['peak_r']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.set_title(key.replace("_"," "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Lag (days)\n← Sent lags FSI  |  Sent leads FSI →", fontsize=9)
        ax.set_ylabel("Pearson r")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8)
        ax.set_xlim(-MAX_LAG-1, MAX_LAG+1)
    plt.suptitle("Cross-Correlation: FSI vs FinBERT Fear Index",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_lead_lag.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  03_lead_lag.png")


def plot_shap(shap_res: dict) -> None:
    by_c = shap_res.get("by_crisis", {})
    if not by_c:
        return
    n = len(by_c)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 7))
    if n == 1:
        axes = [axes]
    pal = [C["crisis"], C["volatile"], C["stable"], C["sentiment"],
           C["fsi"], "#8E44AD", "#16A085", "#D35400"]
    for ax, (crisis, sv) in zip(axes, by_c.items()):
        top = sv.head(8)
        bars = ax.barh(range(len(top)), top.values,
                       color=pal[:len(top)], alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title(crisis.replace("_"," ") + "\nSHAP Attribution",
                     fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        for bar, val in zip(bars, top.values):
            ax.text(val + 5e-4, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=8)
    plt.suptitle("SHAP Feature Importance by Crisis (Random Forest TreeExplainer)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_shap_by_crisis.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  04_shap_by_crisis.png")


def plot_hmm_selection(all_hmm: dict) -> None:
    ns   = sorted(all_hmm.keys())
    bics = [all_hmm[n]["bic"] for n in ns]
    lls  = [all_hmm[n]["ll"]  for n in ns]
    best = ns[int(np.argmin(bics))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(ns, bics, "o-", color=C["crisis"], lw=2, ms=8)
    a1.axvline(best, color="green", ls="--", label=f"Best n={best}")
    a1.set_xticks(ns); a1.set_xlabel("n_states"); a1.set_ylabel("BIC (↓ = better)")
    a1.set_title("HMM Model Selection via BIC", fontweight="bold"); a1.legend()
    a2.plot(ns, lls, "s-", color=C["fsi"], lw=2, ms=8)
    a2.set_xticks(ns); a2.set_xlabel("n_states"); a2.set_ylabel("Log-Likelihood")
    a2.set_title("HMM Log-Likelihood by n_states", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_hmm_selection.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  05_hmm_selection.png")


def plot_garch(feat: pd.DataFrame, all_g: list) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    for ax in axes:
        _shade_crises(ax, alpha=0.08, label=False)

    axes[0].plot(feat["log_ret"]*100, color="#2C3E50", lw=0.4, alpha=0.75)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("Log Return (%)"); axes[0].set_title("S&P 500 Log Returns", fontweight="bold")

    for g in all_g:
        cv = g["cond_vol"]
        if hasattr(cv, "reindex"):
            cv = cv.reindex(feat.index).ffill()
        else:
            cv = pd.Series(cv, index=feat.index[:len(cv)])
        axes[1].plot(feat.index, cv.values if hasattr(cv,"values") else cv,
                     lw=0.7, alpha=0.85, label=g["label"])
    axes[1].set_ylabel("Annualised Vol (σ)")
    axes[1].set_title("GARCH-Family Conditional Volatility Comparison", fontweight="bold")
    axes[1].legend(fontsize=9)

    axes[2].plot(feat["vix"], color=C["garch"], lw=0.8)
    axes[2].axhline(30, color="red", ls="--", alpha=0.5, label="VIX=30")
    axes[2].set_ylabel("VIX"); axes[2].set_xlabel("Date")
    axes[2].set_title("CBOE VIX Index", fontweight="bold"); axes[2].legend(fontsize=9)

    fig.autofmt_xdate(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_garch_all.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  06_garch_all.png")


def plot_fusion_eval(eval_res: dict) -> None:
    rows = []
    for crisis, cr in eval_res.items():
        for model, metrics in cr.items():
            rows.append({"Crisis": crisis.replace("_","\n"),
                         "Model": model, **metrics})
    if not rows:
        return
    df = pd.DataFrame(rows)
    ms  = [m for m in ["f1","prec","rec","auc"] if m in df.columns]
    mls = {"f1":"F1","prec":"Precision","rec":"Recall","auc":"ROC-AUC"}
    fig, axes = plt.subplots(1, len(ms), figsize=(5*len(ms), 5))
    if len(ms) == 1:
        axes = [axes]
    for ax, m in zip(axes, ms):
        pivot = df.pivot(index="Crisis", columns="Model", values=m)
        pivot.plot(kind="bar", ax=ax, width=0.65, colormap="Set2")
        ax.set_title(mls.get(m, m), fontweight="bold")
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
        ax.axhline(FUSION_F1_TARGET, color="red", ls="--", alpha=0.7,
                   label=f"Target ({FUSION_F1_TARGET})")
        ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Fusion Model Evaluation by Crisis Period",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_fusion_eval.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  07_fusion_eval.png")


def plot_research_comparison() -> None:
    data = {
        "Study":           ["Hamilton (1989)","Bollen et al. (2011)",
                            "Riso & Vacca (2024)","Bussmann et al. (2020)",
                            "Ardia et al. (2020)","THIS PROJECT (Group 13)"],
        "Method":          ["HMM","Granger causality","GARCH+NLP",
                            "XAI credit risk","MS-GARCH",
                            "HMM+GARCH+FinBERT+SHAP"],
        "Price signals":   ["✅","❌","✅","✅","✅","✅"],
        "Sentiment":       ["❌","✅","✅","❌","❌","✅"],
        "Explainability":  ["❌","❌","❌","✅","❌","✅"],
        "Explicit lead-lag":["❌","Partial","❌","❌","❌","✅"],
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis("off")
    cc = [["#ECF0F1"]*len(df.columns)]*len(df)
    cc[-1] = ["#D5F5E3"]*len(df.columns)
    t = ax.table(cellText=df.values, colLabels=df.columns,
                 cellLoc="center", loc="center", cellColours=cc)
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.15, 2.3)
    for j in range(len(df.columns)):
        t[(0,j)].set_facecolor("#2C3E50")
        t[(0,j)].set_text_props(color="white", fontweight="bold")
    t[(len(df),0)].set_text_props(fontweight="bold", color="#1A5276")
    ax.set_title("Comparison with Prior Literature (M2 Section 2.2)",
                 fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_research_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  08_research_comparison.png")

# ═══════════════════════════════════════════════════════════════════
# CELL 16 — MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════

def main() -> dict:
    t0 = time.time()
    print("\n" + "╔" + "═"*64 + "╗")
    print("║  MBAI 5600G | Group 13 | Multimodal Financial Crisis Prediction  ║")
    print("╚" + "═"*64 + "╝\n")

    # ── 1. DATA ────────────────────────────────────────────────────
    print("━"*50 + "\n[1/14]  Data acquisition\n" + "━"*50)
    market  = download_all_market()
    fred_df = download_fred()
    news_df = load_news()

    sp500 = market["sp500"]
    vix   = market["vix"]

    # ── 2. FEATURES ────────────────────────────────────────────────
    print("\n" + "━"*50 + "\n[2/14]  Feature engineering\n" + "━"*50)
    feat      = engineer_features(sp500, vix)
    trade_idx = feat.index
    fred_daily = ffill_fred(fred_df, trade_idx)

    # ── 3. FSI (initial, GARCH placeholder = 0) ────────────────────
    print("\n" + "━"*50 + "\n[3/14]  Financial Stress Index (initial)\n" + "━"*50)
    feat, fsi_comps = build_fsi(feat, fred_daily)

    # ── 4. GARCH ───────────────────────────────────────────────────
    print("\n" + "━"*50 + "\n[4/14]  ARMA-GARCH volatility modelling\n" + "━"*50)
    returns          = feat["log_ret"].dropna()
    best_garch, all_garch = select_garch(returns)

    # Extract conditional variance as a pandas Series with correct index
    cond_var = best_garch["cond_var"]           # pandas Series from arch
    if not isinstance(cond_var, pd.Series):
        cond_var = pd.Series(
            cond_var, index=returns.index[:len(cond_var)], name="garch_var")
    cond_var = cond_var.reindex(feat.index).ffill()

    print("\nGARCH Comparison:")
    g_tbl = pd.DataFrame([{k: v for k, v in g.items()
                            if k in ["label","bic","aic","lb_p","arch_p","converged"]}
                           for g in all_garch])
    print(g_tbl.to_string(index=False))

    # ── 5. FSI UPDATE with GARCH ───────────────────────────────────
    print("\n" + "━"*50 + "\n[5/14]  FSI update with GARCH variance\n" + "━"*50)
    feat = update_fsi_garch(feat, fsi_comps, cond_var)

    # ── 6. HMM ─────────────────────────────────────────────────────
    print("\n" + "━"*50 + "\n[6/14]  HMM regime detection\n" + "━"*50)
    best_hmm, all_hmm = select_hmm(feat)
    regime_df         = build_regime_df(best_hmm)
    feat = feat.join(regime_df, how="left")

    # ── 7. FINBERT ─────────────────────────────────────────────────
    print("\n" + "━"*50 + "\n[7/14]  FinBERT sentiment pipeline\n" + "━"*50)
    fb_scores = run_finbert(news_df)
    daily_sent = aggregate_sentiment(fb_scores, trade_idx)

    # Fill coverage gaps with synthetic VIX-based proxy
    if "headline_count" in daily_sent.columns:
        cov = (daily_sent["headline_count"] > 0).mean()
    else:
        cov = 0.0

    if cov < 0.40:
        logger.warning(f"News coverage {cov:.1%} < 40% — merging synthetic proxy")
        synth = build_synthetic_sentiment(feat)
        no_news = (daily_sent.get("headline_count",
                    pd.Series(0, index=daily_sent.index)) == 0)
        for col in ["fear_index","panic_signal","fear_3d","fear_7d","fear_21d"]:
            if col in daily_sent.columns and col in synth.columns:
                daily_sent.loc[no_news, col] = synth.loc[no_news, col]
        daily_sent["is_synthetic"] = no_news.astype(int)

    vader_sent = run_vader(news_df, trade_idx)

    # ── 8. LEAD-LAG ────────────────────────────────────────────────
    print("\n" + "━"*50 + "\n[8/14]  Lead-lag cross-correlation analysis\n" + "━"*50)
    ll_res = run_all_lead_lag(feat, daily_sent)
    gc_df  = run_granger(feat["FSI"], daily_sent["fear_index"])
    print("\nGranger Causality (sentiment → FSI):")
    print(gc_df.to_string(index=False))

    # ── 9. FUSION ──────────────────────────────────────────────────
    print("\n" + "━"*50 + "\n[9/14]  Multimodal fusion model\n" + "━"*50)
    fusion_df = build_fusion(regime_df, daily_sent, feat)
    trained, eval_res = train_models(fusion_df)

    # ── 10. SHAP ───────────────────────────────────────────────────
    print("\n" + "━"*50 + "\n[10/14]  SHAP explainability\n" + "━"*50)
    shap_res, X_shap = run_shap(fusion_df, trained)

    # ── 11. RESEARCH BENCHMARKS ────────────────────────────────────
    print("\n" + "━"*50 + "\n[11/14]  Research paper benchmarks\n" + "━"*50)
    wang_df       = benchmark_wang2025(regime_df)
    fb_vs_vader   = compare_finbert_vader(daily_sent, vader_sent, feat["FSI"])
    print("\nWang et al. (2025) HMM-only baseline:")
    print(wang_df.to_string(index=False))
    if fb_vs_vader:
        print(f"\nFinBERT vs VADER: {fb_vs_vader['interp']}")

    # ── 12. INDIVIDUAL STOCKS ──────────────────────────────────────
    print("\n" + "━"*50 + "\n[12/14]  Cross-sector stock analysis\n" + "━"*50)
    stocks_df = analyse_stocks(market, feat)
    vn_df     = vn_market_robustness()      # optional emerging-market appendix

    # ── 13. VALIDATION CHECKLIST ───────────────────────────────────
    print("\n" + "━"*50 + "\n[13/14]  Crisis validation checklist\n" + "━"*50)
    val_df = validate_checklist(regime_df, daily_sent, eval_res)

    # ── 14. VISUALISATIONS + INTEGRATION CSV ───────────────────────
    print("\n" + "━"*50 + "\n[14/14]  Visualisations & integration CSV\n" + "━"*50)
    plot_regime_timeline(feat, regime_df)
    plot_sentiment_fsi(feat, daily_sent)
    plot_lead_lag(ll_res)
    plot_shap(shap_res)
    plot_hmm_selection(all_hmm)
    plot_garch(feat, all_garch)
    plot_fusion_eval(eval_res)
    plot_research_comparison()

    # Integration CSV (M3 shared interface) — richer, with provenance flags
    keep = [c for c in ["regime","prob_stable","prob_volatile","prob_crisis","FSI"]
            if c in feat.columns]
    integ = feat[keep].copy()
    for col in ["fear_index","fear_3d","fear_7d","panic_signal",
                "headline_count","is_synthetic"]:
        if col in daily_sent.columns:
            integ[col] = daily_sent[col].reindex(integ.index)
    integ.to_csv(OUTPUT_DIR / "integration_master.csv")

    # Machine-readable metrics summary (handy for the report / grading)
    best_f1 = {c: round(max(m["f1"] for m in cr.values()), 4)
               for c, cr in eval_res.items()}
    metrics = {
        "best_garch": best_garch["label"], "best_garch_bic": round(best_garch["bic"], 2),
        "hmm_n_retained": best_hmm["model"].n_components,
        "hmm_bic_profile": {n: round(all_hmm[n]["bic"], 2) for n in all_hmm},
        "fsi_validity": feat.attrs.get("fsi_validity", {}),
        "lead_lag": {k: {"peak_lag": v["peak_lag"], "peak_r": round(v["peak_r"], 4),
                         "interp": v["interp"]} for k, v in ll_res.items()},
        "fusion_best_f1_by_crisis": best_f1,
        "fusion_f1_target": FUSION_F1_TARGET,
        "finbert_vs_vader": fb_vs_vader,
    }
    with open(OUTPUT_DIR / "metrics_summary.json", "w") as jf:
        json.dump(metrics, jf, indent=2, default=str)

    elapsed = time.time() - t0

    # ── FINAL SUMMARY ──────────────────────────────────────────────
    print("\n" + "╔" + "═"*64 + "╗")
    print("║                   PIPELINE COMPLETE                          ║")
    print("╚" + "═"*64 + "╝")
    print(f"\n  Runtime   : {elapsed/60:.1f} minutes")
    print(f"  Best GARCH: {best_garch['label']}  BIC={best_garch['bic']:.2f}")
    print(f"  HMM       : n={best_hmm['model'].n_components} retained  "
          f"(BIC profile { {n: round(all_hmm[n]['bic']) for n in all_hmm} })")
    fv = feat.attrs.get("fsi_validity", {})
    print(f"  FSI       : STLFSI r={fv.get('stlfsi_r')}  "
          f"NBER-AUC={fv.get('nber_roc_auc')}  (target r>{FSI_CORR_TARGET})")
    print(f"  Lead-lag  : {ll_res['overall']['interp']}  "
          f"(r={ll_res['overall']['peak_r']:.4f})")
    print(f"  Fusion F1 : {best_f1}  (target ≥ {FUSION_F1_TARGET})")
    if fb_vs_vader:
        print(f"  NLP bench : {fb_vs_vader['interp']}")
    print(f"\n  Integration CSV: {len(integ):,} rows")
    print("\n  Output files:")
    for f_ in sorted(OUTPUT_DIR.glob("*.*")):
        if not f_.is_dir():
            print(f"    {f_.name}  ({f_.stat().st_size/1024:.0f} KB)")

    return dict(
        feat=feat, regime_df=regime_df, daily_sent=daily_sent,
        fusion_df=fusion_df, trained=trained, eval_res=eval_res,
        shap_res=shap_res, ll_res=ll_res, val_df=val_df,
        best_garch=best_garch, best_hmm=best_hmm,
        all_hmm=all_hmm, all_garch=all_garch,
        stocks_df=stocks_df, vn_df=vn_df, wang_df=wang_df, gc_df=gc_df,
        metrics=metrics,
    )


# ═══════════════════════════════════════════════════════════════════
# CELL 17 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = main()

    # ── Notebook-level convenience handles ─────────────────────────
    feat         = results["feat"]
    regime_df    = results["regime_df"]
    daily_sent   = results["daily_sent"]
    fusion_df    = results["fusion_df"]
    shap_res     = results["shap_res"]
    val_df       = results["val_df"]
    ll_res       = results["ll_res"]
    eval_res     = results["eval_res"]

    print("\n✅  All results saved to /kaggle/working/outputs/")
    print("    Access any result with: results['<key>']")
    print("\n    Available keys:", list(results.keys()))