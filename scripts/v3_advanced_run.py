#!/usr/bin/env python3
"""
v3 ADVANCED — calibration + VIX-orthogonal macro features + walk-forward-everything.

Brings together roadmap items 1, 3, 4 (and the optional real-news ablation, item 2)
on the honest harness:
  - Probability calibration inside each walk-forward fold (src/v3/calibration.py).
  - VIX-orthogonal macro features: credit, yield curve, funding, cross-asset corr.
  - Online (per-fold refit) regime probabilities (src/v3/online_features.py).
  - Optional: real FinBERT news sentiment ablation if NEWS_DATA_DIR is set.

Author: Romin Patel.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 21
DD_THRESHOLD = 0.10


def main() -> dict:
    print("=" * 76)
    print("  FCPS v3 ADVANCED — calibration + macro(VIX-orthogonal) + online regime")
    print("=" * 76)

    from src.config import cfg
    from src.logging_setup import setup_logging
    setup_logging(level="WARNING", fmt="text")

    from src.data.market import download_all_market
    from src.data.fred import download_fred, align_fred_to_trading_days
    from src.features.engineering import engineer_features
    from src.models.fsi import FSIBuilder
    from src.v3.labeling import crisis_label, label_summary
    from src.v3.macro_features import build_macro_features, usable_macro_cols
    from src.v3.online_features import compute_online_regime
    from src.v3.baselines import VixThresholdPredictor
    from src.v3.calibration import make_calibrated_factory
    from src.v3.walkforward import WalkForwardConfig, walk_forward_predict
    from src.v3.metrics import classification_metrics

    # ── Data + base features ─────────────────────────────────────────
    print("\n[1] Data + features + FSI (train-head scaler)...")
    market = download_all_market()
    feat = engineer_features(market["sp500"], market["vix"])
    n = len(feat)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[: int(n * 0.5)] = True
    feat, _ = FSIBuilder().build(feat, pd.DataFrame(), train_mask=train_mask)
    fred = download_fred()
    fred_daily = align_fred_to_trading_days(fred, feat.index)

    # ── Online (walk-forward refit) regime ───────────────────────────
    print("[2] Online regime (per-fold refit HMM+scaler, causal)...")
    online = compute_online_regime(feat, ["log_ret", "vol_21d", "FSI"],
                                    min_train=1260, refit_step=378, n_seeds=5, n_iter=80)
    feat = feat.join(online, how="left")

    # ── VIX-orthogonal macro features ────────────────────────────────
    print("[3] VIX-orthogonal macro features...")
    macro = build_macro_features(fred_daily, market, feat.index)
    feat = feat.join(macro, how="left")

    # ── Optional real-news FinBERT sentiment (roadmap item 2) ────────
    real_sent_col = None
    news_dir = os.environ.get("NEWS_DATA_DIR")
    if news_dir:
        print(f"[3b] NEWS_DATA_DIR set ({news_dir}) — running REAL FinBERT sentiment...")
        try:
            from src.data.news import load_news
            from src.models.sentiment import run_finbert, aggregate_sentiment
            news = load_news(Path(news_dir))
            fb = run_finbert(news)
            daily = aggregate_sentiment(fb, feat.index)
            feat["fear_real"] = daily["fear_index"].reindex(feat.index)
            real_sent_col = "fear_real"
            print("      Real FinBERT sentiment merged as 'fear_real'.")
        except Exception as exc:
            print(f"      Real-news ablation failed ({exc}); continuing without it.")
    else:
        print("[3b] NEWS_DATA_DIR not set — real-news FinBERT ablation SKIPPED.")
        print("      To run it:  NEWS_DATA_DIR=/path/to/news/csvs python scripts/v3_advanced_run.py")

    # ── Label ────────────────────────────────────────────────────────
    label = crisis_label(feat["close"], horizon=HORIZON, drawdown_threshold=DD_THRESHOLD)
    print(f"\n[4] Exogenous label (fwd {HORIZON}d drawdown <= -{DD_THRESHOLD:.0%}): {label_summary(label)}")

    # ── Feature groups ───────────────────────────────────────────────
    price_cols = [c for c in ["vol_21d", "vix", "drawdown_63", "mom_21d", "mom_63d", "vix_spike", "vol_ratio"] if c in feat.columns]
    macro_cols = usable_macro_cols(feat)
    online_cols = [c for c in ["o_prob_volatile", "o_prob_crisis"] if c in feat.columns]

    all_cols = price_cols + macro_cols + online_cols + ([real_sent_col] if real_sent_col else [])
    F = feat[all_cols].copy()
    F["label"] = label.reindex(F.index)
    F = F.dropna(subset=all_cols)  # honest: drop early rows lacking credit-spread etc.
    y = F["label"]
    print(f"    Usable rows after dropna: {len(F):,}  ({F.index.min().date()} -> {F.index.max().date()})")

    wf = WalkForwardConfig(min_train=1260, step=126, embargo=HORIZON, horizon=HORIZON)

    def lr():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000, random_state=cfg.seed))

    def rf():
        return RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=20,
                                      class_weight="balanced", n_jobs=-1, random_state=cfg.seed)

    full_cols = price_cols + macro_cols + online_cols + ([real_sent_col] if real_sent_col else [])

    contenders = {
        "BASELINE VIX-threshold":          (VixThresholdPredictor, price_cols),
        "LR price-only":                   (lr, price_cols),
        "LR price+macro":                  (lr, price_cols + macro_cols),
        "LR price+macro+online":           (lr, full_cols),
        "LR full CALIBRATED":              (make_calibrated_factory(lr, method="isotonic"), full_cols),
        "RF full CALIBRATED":              (make_calibrated_factory(rf, method="isotonic"), full_cols),
    }

    print("\n[5] Walk-forward (expanding, quarterly refit, 21d embargo)\n")
    print(f"  {'Model':28} {'PR-AUC':>7} {'BSS':>8} {'ECE':>7} {'lift@10%':>9}")
    print("  " + "-" * 64)
    results = {}
    for name, (factory, cols) in contenders.items():
        res = walk_forward_predict(F[cols], y, factory, wf)
        m = classification_metrics(res.oos_label.to_numpy(dtype=float), res.oos_proba.to_numpy(dtype=float))
        results[name] = m
        print(f"  {name:28} {str(m['pr_auc']):>7} {str(m['brier_skill']):>8} {str(m['ece']):>7} {str(m['lift_top_decile']):>9}")

    # ── Save + verdict ───────────────────────────────────────────────
    from src.json_utils import safe_json_default
    out = {
        "task": f"crisis (fwd {HORIZON}d drawdown<=-{DD_THRESHOLD:.0%}) — advanced",
        "label_summary": label_summary(label),
        "feature_groups": {"price": price_cols, "macro_vix_orthogonal": macro_cols,
                           "online_regime": online_cols, "real_sentiment": real_sent_col},
        "walkforward": wf.__dict__,
        "metrics": results,
        "real_news_ablation_ran": real_sent_col is not None,
    }
    with open(cfg.paths.output_dir / "v3_advanced_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, default=safe_json_default)

    print("\n" + "=" * 76)
    print("  VERDICT")
    print("=" * 76)
    uncal = results.get("LR price+macro+online", {})
    cal = results.get("LR full CALIBRATED", {})
    vix = results.get("BASELINE VIX-threshold", {})

    def _f(x):
        return x if isinstance(x, (int, float)) and x is not None and not (isinstance(x, float) and np.isnan(x)) else None

    print(f"  Calibration effect (LR full):  BSS {_f(uncal.get('brier_skill'))} -> {_f(cal.get('brier_skill'))} | "
          f"ECE {_f(uncal.get('ece'))} -> {_f(cal.get('ece'))}")
    best_ml = max((results[k]['pr_auc'] for k in results if k != 'BASELINE VIX-threshold'
                   and isinstance(results[k]['pr_auc'], float) and not np.isnan(results[k]['pr_auc'])), default=float('nan'))
    print(f"  Best ML PR-AUC {round(best_ml,4)} vs VIX baseline {vix.get('pr_auc')} -> "
          f"{'macro/regime add value' if (best_ml or 0) > (vix.get('pr_auc') or 0) else 'still no decisive edge over VIX'}")
    print(f"\n  Saved: {cfg.paths.output_dir / 'v3_advanced_metrics.json'}")
    return out


if __name__ == "__main__":
    main()
