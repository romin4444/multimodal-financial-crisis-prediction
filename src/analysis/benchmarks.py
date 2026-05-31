"""
Research paper comparison benchmarks.
Wang et al. (2025) HMM-only baseline and FinBERT vs VADER comparison.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
from scipy import stats

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("analysis.benchmarks")


def benchmark_wang2025(regime: pd.DataFrame) -> pd.DataFrame:
    """
    Wang et al. (2025): HMM-only early-warning lead time in trading days.
    Detection = HMM enters crisis state within ±10 trading days of onset.
    """
    rows = []
    for crisis_name, (cs, ce) in cfg.crisis_window_tuples.items():
        first_date, lead_td, detected = _trading_lead(regime, pd.Timestamp(cs))
        rows.append({
            "Crisis": crisis_name,
            "Period": f"{cs} → {ce}",
            "Detected": "✅" if detected else "❌",
            "First_crisis_state": str(first_date.date()) if first_date is not None else "N/A",
            "Lead_trading_days": lead_td,
            "By_onset_plus_10td": "✅" if detected else "❌",
        })
    df = pd.DataFrame(rows)
    log.info("Wang2025 benchmark complete", extra={"results": rows})
    return df


def compare_finbert_vader(
    sent_fb: pd.DataFrame,
    sent_vader: pd.DataFrame,
    fsi: pd.Series,
) -> dict:
    """
    Pearson correlation of FinBERT and VADER fear indices with FSI.
    Shobayo et al. (2024): FinBERT outperforms VADER on crisis text.
    """
    if sent_vader.empty or "vader_fear" not in sent_vader.columns:
        log.warning("VADER data unavailable — skipping comparison")
        return {}

    idx = fsi.index.intersection(sent_fb.index).intersection(sent_vader.index)
    f0 = fsi.reindex(idx).fillna(0)
    fb = sent_fb["fear_index"].reindex(idx).fillna(0)
    vd = sent_vader["vader_fear"].reindex(idx).fillna(0)

    r_fb, p_fb = stats.pearsonr(f0, fb)
    r_vd, p_vd = stats.pearsonr(f0, vd)
    winner = "FinBERT" if abs(r_fb) > abs(r_vd) else "VADER"

    result = dict(
        finbert_r=round(float(r_fb), 4),
        finbert_p=round(float(p_fb), 4),
        vader_r=round(float(r_vd), 4),
        vader_p=round(float(p_vd), 4),
        winner=winner,
        interp=f"{winner} wins | FinBERT r={r_fb:.4f}  VADER r={r_vd:.4f}",
    )
    log.info("FinBERT vs VADER", extra=result)
    return result


# ─── Private ─────────────────────────────────────────────────────────────────

def _trading_lead(
    regime: pd.DataFrame,
    onset: pd.Timestamp,
    lookback_td: int = 60,
    lookahead_td: int = 10,
) -> Tuple[Optional[pd.Timestamp], Optional[int], bool]:
    """
    Find first crisis-state day within [onset - lookback_td, onset + lookahead_td] td.
    Returns (first_date, lead_days, detected).
    Positive lead = detected BEFORE onset (early warning).
    """
    idx = regime.index
    pos = idx.searchsorted(onset)
    lo = max(0, pos - lookback_td)
    hi = min(len(idx) - 1, pos + lookahead_td)
    window = regime.iloc[lo: hi + 1]
    crisis_days = window.index[window["regime"] == 2]

    if len(crisis_days) == 0:
        return None, None, False

    first = crisis_days[0]
    lead_td = int(pos - idx.searchsorted(first))
    return first, lead_td, True
