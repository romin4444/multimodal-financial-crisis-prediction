"""
Crisis validation checklist (M2 Section 4.5).
Req 1: HMM crisis state within ±10 trading days of onset.
Req 2: Panic signal fires 30 calendar days before onset.
Req 3: Best held-out F1 ≥ 0.70.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.config import cfg
from src.logging_setup import get_logger
from src.models.fusion import ModelEvaluation

log = get_logger("evaluation.validation")


def validate_checklist(
    regime: pd.DataFrame,
    sent: pd.DataFrame,
    evaluations: Dict[str, List[ModelEvaluation]],
) -> pd.DataFrame:
    """
    Run all three M2 validation requirements for each crisis window.
    Returns a summary DataFrame with pass/fail per requirement.
    """
    rows = []
    for crisis_name, (cs, ce) in cfg.crisis_window_tuples.items():
        start = pd.Timestamp(cs)

        # Req 1 — HMM crisis state detected within ±10 trading days
        req1, first_date, lead_td = _check_hmm_detection(regime, start)

        # Req 2 — panic signal fires in 30 calendar days before onset
        req2 = _check_panic_signal(sent, start)

        # Req 3 — best held-out F1 ≥ target
        best_f1, req3 = _check_fusion_f1(evaluations, crisis_name)

        all_pass = req1 and (req2 is not False) and req3
        rows.append({
            "Crisis": crisis_name,
            "Period": f"{cs} → {ce}",
            "Req1_HMM_detection": "✅" if req1 else "❌",
            "First_detected": str(first_date.date()) if first_date else "—",
            "Lead_td": lead_td if lead_td is not None else "—",
            "Req2_Panic_before": ("✅" if req2 else "❌") if req2 is not None else "—",
            "Best_F1": f"{best_f1:.4f}" if best_f1 is not None else "—",
            "Req3_F1_target": "✅" if req3 else "❌",
            "All_pass": "✅" if all_pass else "❌",
        })
        log.info(
            "Validation checklist",
            extra={
                "crisis": crisis_name,
                "req1": req1, "req2": req2, "req3": req3,
                "all_pass": all_pass, "best_f1": best_f1,
            },
        )

    df = pd.DataFrame(rows)
    n_pass = (df["All_pass"] == "✅").sum()
    log.info(
        "Validation summary",
        extra={"crises_all_pass": int(n_pass), "total_crises": len(df)},
    )
    return df


# ─── Private ─────────────────────────────────────────────────────────────────

def _check_hmm_detection(
    regime: pd.DataFrame, onset: pd.Timestamp, lookahead_td: int = 10, lookback_td: int = 60
):
    idx = regime.index
    pos = idx.searchsorted(onset)
    lo = max(0, pos - lookback_td)
    hi = min(len(idx) - 1, pos + lookahead_td)
    window = regime.iloc[lo: hi + 1]
    crisis_days = window.index[window["regime"] == 2]
    if len(crisis_days) == 0:
        return False, None, None
    first = crisis_days[0]
    lead_td = int(pos - idx.searchsorted(first))
    return True, first, lead_td


def _check_panic_signal(
    sent: pd.DataFrame, onset: pd.Timestamp, lookback_days: int = 30
) -> Optional[bool]:
    if "panic_signal" not in sent.columns:
        return None
    pre = sent[
        (sent.index >= onset - pd.Timedelta(days=lookback_days))
        & (sent.index < onset)
    ]
    return bool((pre["panic_signal"] == 1).any())


def _check_fusion_f1(
    evaluations: Dict[str, List[ModelEvaluation]], crisis_name: str
) -> tuple:
    if crisis_name not in evaluations:
        return None, False
    evals = evaluations[crisis_name]
    if not evals:
        return None, False
    best_f1 = max(ev.f1 for ev in evals)
    return best_f1, best_f1 >= cfg.fusion.f1_target
