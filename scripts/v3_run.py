#!/usr/bin/env python3
"""
v3 honest backtest — leakage-free, walk-forward, exogenous-label evaluation.

Demonstrates on REAL S&P 500 + VIX (cached) that:
  1. With an exogenous forward-drawdown target (not the HMM's own label),
  2. causal (filtered) regime probabilities (no future smoothing),
  3. and purged walk-forward evaluation (no peeking at crisis windows),
the honest out-of-sample skill is far below v2's in-sample F1 = 0.99 — and we
measure whether sentiment/regime features add anything over price + VIX alone.
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
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> dict:
    print("=" * 72)
    print("  FCPS v3 — HONEST WALK-FORWARD BACKTEST (real S&P 500 + VIX)")
    print("=" * 72)

    from src.config import cfg
    from src.logging_setup import setup_logging
    setup_logging(level="WARNING", fmt="text")  # quiet — we print our own table

    from src.data.market import download_all_market
    from src.features.engineering import engineer_features
    from src.models.fsi import FSIBuilder
    from src.v3.labeling import crisis_label, label_summary
    from src.v3.causal_regime import causal_regime_frame
    from src.v3.baselines import BaseRatePredictor, VixThresholdPredictor, PersistencePredictor
    from src.v3.walkforward import WalkForwardConfig, walk_forward_predict
    from src.v3.metrics import classification_metrics, economic_backtest

    HORIZON = 21
    DD_THRESHOLD = 0.10
    TRAIN_HEAD_FRAC = 0.50

    # ── Data + features ──────────────────────────────────────────────
    print("\n[1] Loading real market data + engineering features...")
    market = download_all_market()
    feat = engineer_features(market["sp500"], market["vix"])

    n = len(feat)
    train_head = int(n * TRAIN_HEAD_FRAC)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_head] = True

    # FSI with scaler fit on train-head only (leakage-free)
    fsi_builder = FSIBuilder()
    feat, _ = fsi_builder.build(feat, pd.DataFrame(), train_mask=train_mask)
    print(f"    {n:,} trading days  {feat.index.min().date()} -> {feat.index.max().date()}")

    # ── Exogenous label ──────────────────────────────────────────────
    print(f"\n[2] Building EXOGENOUS label: forward {HORIZON}d drawdown <= -{DD_THRESHOLD:.0%}")
    label = crisis_label(feat["close"], horizon=HORIZON, drawdown_threshold=DD_THRESHOLD)
    print(f"    {label_summary(label)}")

    # ── Causal (filtered) regime probabilities ───────────────────────
    print("\n[3] Fitting HMM on train-head, computing CAUSAL filtered posteriors...")
    hmm_feats = [c for c in ["log_ret", "vol_21d", "FSI"] if c in feat.columns]
    Xh = feat[hmm_feats].dropna()
    scaler = StandardScaler().fit(Xh.iloc[: int(len(Xh) * TRAIN_HEAD_FRAC)])
    Xh_scaled = scaler.transform(Xh)

    best_hmm, best_ll = None, -np.inf
    for seed in range(12):
        try:
            m = GaussianHMM(n_components=3, covariance_type="full", n_iter=150, random_state=seed)
            m.fit(Xh_scaled[: int(len(Xh) * TRAIN_HEAD_FRAC)])  # fit on train-head ONLY
            ll = m.score(Xh_scaled[: int(len(Xh) * TRAIN_HEAD_FRAC)])
            if ll > best_ll:
                best_ll, best_hmm = ll, m
        except Exception:
            pass
    regime = causal_regime_frame(best_hmm, Xh_scaled, Xh.index, hmm_feats)
    feat = feat.join(regime, how="left")
    print(f"    Causal regime mix: "
          f"stable={ (regime['c_regime']==0).mean():.1%} "
          f"volatile={ (regime['c_regime']==1).mean():.1%} "
          f"crisis={ (regime['c_regime']==2).mean():.1%}")

    # ── Sentiment (synthetic, VIX-derived — flagged as confounded) ───
    from src.models.sentiment import build_synthetic_sentiment
    sent = build_synthetic_sentiment(feat.dropna(subset=["log_ret", "vix"]))
    feat["fear_index"] = sent["fear_index"].reindex(feat.index)

    # ── Assemble model matrix ────────────────────────────────────────
    price_cols = [c for c in ["vol_21d", "vix", "drawdown_63", "mom_21d", "mom_63d", "vix_spike", "vol_ratio"] if c in feat.columns]
    regime_cols = [c for c in ["c_prob_volatile", "c_prob_crisis"] if c in feat.columns]
    sent_cols = ["fear_index"]

    F = feat[price_cols + regime_cols + sent_cols].copy()
    F["label"] = label.reindex(F.index)
    F = F.dropna(subset=price_cols)  # keep rows with valid features
    y = F["label"]

    wf = WalkForwardConfig(min_train=1260, step=63, embargo=HORIZON, horizon=HORIZON)

    # ── Define contenders ────────────────────────────────────────────
    def logistic():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000, random_state=cfg.seed))

    def rf():
        return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                      class_weight="balanced", n_jobs=-1, random_state=cfg.seed)

    contenders = {
        "BASELINE base-rate":      (BaseRatePredictor, price_cols),
        "BASELINE VIX-threshold":  (VixThresholdPredictor, price_cols),
        "BASELINE persistence":    (PersistencePredictor, price_cols),
        "MODEL price-only (LR)":   (logistic, price_cols),
        "MODEL +regime (LR)":      (logistic, price_cols + regime_cols),
        "MODEL +regime+sent (LR)": (logistic, price_cols + regime_cols + sent_cols),
        "MODEL full (RandomForest)": (rf, price_cols + regime_cols + sent_cols),
    }

    print("\n[4] Walk-forward (expanding window, quarterly refit, 21d embargo)...")
    results = {}
    oos_store = {}
    for name, (factory, cols) in contenders.items():
        Xc = F[cols]
        res = walk_forward_predict(Xc, y, factory, wf)
        m = classification_metrics(res.oos_label.to_numpy(dtype=float),
                                    res.oos_proba.to_numpy(dtype=float))
        results[name] = m
        oos_store[name] = res.oos_proba
        print(f"    {name:28} folds={res.n_folds:>2}  PR-AUC={m['pr_auc']}  BSS={m['brier_skill']}  lift@10%={m['lift_top_decile']}")

    # ── Economic backtest on best model ──────────────────────────────
    print("\n[5] Economic backtest (de-risk when crisis prob is high)...")
    fwd_ret = feat["close"].pct_change().shift(-1)  # return t->t+1, decided at t
    best_name = max(
        (k for k in results if k.startswith("MODEL")),
        key=lambda k: (results[k]["pr_auc"] if not np.isnan(results[k]["pr_auc"]) else -1),
    )
    best_oos = oos_store[best_name].dropna()
    thr = float(best_oos.quantile(0.85))  # de-risk on riskiest ~15% of days
    econ = economic_backtest(fwd_ret, best_oos, threshold=thr)
    print(f"    Best model: {best_name} (de-risk threshold p>{thr:.3f})")
    for k, v in econ.items():
        print(f"      {k:28}: {v}")

    # ── Save + verdict ───────────────────────────────────────────────
    out = {
        "config": {"horizon_days": HORIZON, "drawdown_threshold": DD_THRESHOLD,
                   "train_head_frac": TRAIN_HEAD_FRAC,
                   "walkforward": wf.__dict__},
        "label_summary": label_summary(label),
        "metrics": results,
        "best_model": best_name,
        "economic_backtest": econ,
        "note": "EXOGENOUS forward-drawdown label; CAUSAL filtered regime probs; "
                "purged walk-forward OOS. Compare to v2 in-sample F1=0.99.",
    }
    def _safe(o):
        import math
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return str(o)
    with open(cfg.paths.output_dir / "v3_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, default=_safe)

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    pa = results["MODEL +regime+sent (LR)"]["pr_auc"]
    base = results["BASELINE VIX-threshold"]["pr_auc"]
    print("  v2 reported (in-sample, leaky)        : F1 = 0.99")
    print(f"  v3 honest OOS (full model PR-AUC)     : {pa}")
    print(f"  v3 VIX-only baseline PR-AUC           : {base}")
    print(f"  -> sentiment/regime add value?         {'YES' if (pa or 0) > (base or 0) else 'MARGINAL/NO'}")
    print(f"\n  Saved: {cfg.paths.output_dir / 'v3_metrics.json'}")
    return out


if __name__ == "__main__":
    main()
