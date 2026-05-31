#!/usr/bin/env python3
"""
v3 — Discrete-time HAZARD / survival model run.

Reframes crisis onset as time-to-event: estimates the per-day hazard of entering
a >=10% drawdown and converts it to P(>=10% drawdown within N days). Reports
concordance (C-index) and N-day cumulative-incidence calibration on a held-out
temporal test split.

Author: Romin Patel.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

DD_THRESHOLD = 0.10
HORIZONS = [21, 63]


def main() -> dict:
    print("=" * 72)
    print("  FCPS v3 — DISCRETE-TIME HAZARD / SURVIVAL MODEL")
    print("=" * 72)

    from src.config import cfg
    from src.logging_setup import setup_logging
    setup_logging(level="WARNING", fmt="text")

    from src.data.market import download_all_market
    from src.data.fred import download_fred, align_fred_to_trading_days
    from src.features.engineering import engineer_features
    from src.models.fsi import FSIBuilder
    from src.v3.macro_features import build_macro_features, usable_macro_cols
    from src.v3.hazard import drawdown_panel, fit_hazard, evaluate_hazard

    print("\n[1] Data + features...")
    market = download_all_market()
    feat = engineer_features(market["sp500"], market["vix"])
    n = len(feat)
    train_mask = np.zeros(n, dtype=bool); train_mask[: int(n * 0.6)] = True
    feat, _ = FSIBuilder().build(feat, pd.DataFrame(), train_mask=train_mask)
    fred_daily = align_fred_to_trading_days(download_fred(), feat.index)
    macro = build_macro_features(fred_daily, market, feat.index)
    feat = feat.join(macro, how="left")

    print("[2] Building survival panel (drawdown onsets / at-risk / duration)...")
    panel = drawdown_panel(feat["close"], threshold=DD_THRESHOLD)
    n_onsets = int(panel["onset"].sum())
    print(f"    >=10% drawdown episodes: {n_onsets}  | at-risk days: {int(panel['at_risk'].sum()):,}")

    # Features for hazard: price stress + well-populated VIX-orthogonal macro
    feat_cols = [c for c in ["vol_21d", "vix", "drawdown_63", "mom_21d"] if c in feat.columns]
    feat_cols += usable_macro_cols(feat)

    # Temporal split: train on first 60%, test on last 40%
    idx = feat.index
    cut = int(len(idx) * 0.6)
    tr_mask = np.zeros(len(idx), dtype=bool); tr_mask[:cut] = True
    te_mask = ~tr_mask
    print(f"[3] Train {idx[0].date()}->{idx[cut-1].date()} | Test {idx[cut].date()}->{idx[-1].date()}")

    fit = fit_hazard(panel, feat, feat_cols, tr_mask)

    print("\n[4] Out-of-sample hazard evaluation:\n")
    print(f"  {'Horizon':>8} {'C-index':>9} {'Nday base':>10} {'Nday Brier':>11} {'Brier skill':>12}")
    print("  " + "-" * 54)
    results = {}
    for h in HORIZONS:
        m = evaluate_hazard(fit, panel, feat, horizon=h, test_mask=te_mask, drawdown_threshold=DD_THRESHOLD)
        results[f"h{h}"] = m
        print(f"  {h:>8} {str(m.get('c_index')):>9} {str(m.get('Nday_base_rate')):>10} "
              f"{str(m.get('Nday_risk_brier')):>11} {str(m.get('Nday_risk_brier_skill')):>12}")

    def _safe(o):
        import math
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return str(o)
    out = {
        "task": "discrete-time hazard: P(>=10% drawdown within N days)",
        "drawdown_threshold": DD_THRESHOLD,
        "n_onsets": n_onsets,
        "feature_cols": fit.feature_cols,
        "results": results,
    }
    with open(cfg.paths.output_dir / "hazard_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, default=_safe)

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    c21 = results.get("h21", {}).get("c_index")
    print(f"  Hazard concordance (C-index, 21d): {c21}  "
          f"({'better than chance' if isinstance(c21, float) and c21 > 0.55 else 'near chance'})")
    print(f"  Saved: {cfg.paths.output_dir / 'hazard_metrics.json'}")
    return out


if __name__ == "__main__":
    main()
