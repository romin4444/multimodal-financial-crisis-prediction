#!/usr/bin/env python3
"""
v3 — Multi-stock DIRECTION (up/down) detection backtest.

Honest, leakage-free walk-forward evaluation of next-N-day direction prediction
for AAPL, JPM, XOM, GS and the S&P 500. Every model is benchmarked against the
always-up, momentum, and base-rate baselines. The question we answer is the only
one that matters: does any model beat 'always predict up' OUT-OF-SAMPLE?
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

HORIZON = 5  # predict direction over next 5 trading days (weekly)


def build_market_context() -> pd.DataFrame:
    """Index-wide causal features used as context for every stock."""
    from src.features.engineering import engineer_features
    from src.models.fsi import FSIBuilder
    from src.v3.causal_regime import causal_regime_frame
    from src.data.market import download_all_market

    market = download_all_market()
    feat = engineer_features(market["sp500"], market["vix"])

    n = len(feat)
    train_head = int(n * 0.5)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_head] = True

    fsi_builder = FSIBuilder()
    feat, _ = fsi_builder.build(feat, pd.DataFrame(), train_mask=train_mask)

    hmm_feats = [c for c in ["log_ret", "vol_21d", "FSI"] if c in feat.columns]
    Xh = feat[hmm_feats].dropna()
    head = int(len(Xh) * 0.5)
    scaler = StandardScaler().fit(Xh.iloc[:head])
    Xh_scaled = scaler.transform(Xh)

    best_hmm, best_ll = None, -np.inf
    for seed in range(10):
        try:
            m = GaussianHMM(n_components=3, covariance_type="full", n_iter=120, random_state=seed)
            m.fit(Xh_scaled[:head])
            ll = m.score(Xh_scaled[:head])
            if ll > best_ll:
                best_ll, best_hmm = ll, m
        except Exception:
            pass
    regime = causal_regime_frame(best_hmm, Xh_scaled, Xh.index, hmm_feats)

    ctx = pd.DataFrame(index=feat.index)
    ctx["mkt_vix"] = feat["vix"]
    ctx["mkt_fsi"] = feat["FSI"]
    ctx["mkt_mom_21"] = feat["close"].pct_change(21)
    ctx["mkt_prob_volatile"] = regime["c_prob_volatile"].reindex(feat.index)
    ctx["mkt_prob_crisis"] = regime["c_prob_crisis"].reindex(feat.index)
    return market, ctx


def main() -> dict:
    print("=" * 74)
    print(f"  FCPS v3 — STOCK DIRECTION DETECTION ({HORIZON}-day, walk-forward, honest)")
    print("=" * 74)

    from src.config import cfg
    from src.logging_setup import setup_logging
    setup_logging(level="WARNING", fmt="text")

    from src.v3.direction import (
        build_stock_features, direction_label, direction_summary,
        STOCK_FEATURE_COLS, MARKET_CONTEXT_COLS,
        AlwaysUpPredictor, MomentumPredictor, directional_metrics, directional_backtest,
    )
    from src.v3.baselines import BaseRatePredictor
    from src.v3.walkforward import WalkForwardConfig, walk_forward_predict

    print("\n[1] Building market context (causal index regime + VIX + FSI)...")
    market, ctx = build_market_context()

    tickers = {"AAPL": "aapl", "JPM": "jpm", "XOM": "xom", "GS": "gs", "^GSPC": "sp500"}
    wf = WalkForwardConfig(min_train=1000, step=126, embargo=HORIZON, horizon=HORIZON)

    def logistic():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=cfg.seed),
        )

    def rf():
        return RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_leaf=25,
            class_weight="balanced", n_jobs=-1, random_state=cfg.seed,
        )

    all_results: dict = {}
    print(f"\n[2] Walk-forward per stock (min_train={wf.min_train}, step={wf.step}, embargo={wf.embargo})\n")
    header = f"  {'Stock':6} {'Model':22} {'Acc':>7} {'Majority':>9} {'Edge':>7} {'AUC':>7} {'MCC':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for tk, key in tickers.items():
        if key not in market:
            continue
        stk = market[key]
        feats = build_stock_features(stk, ctx)
        label = direction_label(stk["Close"], horizon=HORIZON)

        price_cols = [c for c in STOCK_FEATURE_COLS if c in feats.columns]
        ctx_cols = [c for c in MARKET_CONTEXT_COLS if c in feats.columns]

        F = feats[price_cols + ctx_cols].copy()
        F["label"] = label.reindex(F.index)
        F = F.dropna(subset=price_cols)
        y = F["label"]

        contenders = {
            "BASE always-up": (AlwaysUpPredictor, price_cols),
            "BASE momentum": (MomentumPredictor, price_cols),
            "BASE base-rate": (BaseRatePredictor, price_cols),
            "LR price-only": (logistic, price_cols),
            "LR price+market": (logistic, price_cols + ctx_cols),
            "RF full": (rf, price_cols + ctx_cols),
        }

        stock_res = {"direction_summary": direction_summary(label), "models": {}, "oos": {}}
        for name, (factory, cols) in contenders.items():
            res = walk_forward_predict(F[cols], y, factory, wf)
            m = directional_metrics(res.oos_label.to_numpy(dtype=float),
                                    res.oos_proba.to_numpy(dtype=float))
            stock_res["models"][name] = m
            stock_res["oos"][name] = res.oos_proba
            print(f"  {tk:6} {name:22} {m['accuracy']:>7} {m['majority_acc']:>9} "
                  f"{m['edge_over_majority']:>+7} {str(m['auc']):>7} {str(m['mcc']):>7}")
        print()
        all_results[tk] = stock_res

    # ── Economic backtest: best model per stock (long/flat) ──────────
    print("[3] Economic backtest (long when P(up)>=0.5 else cash) — best model per stock\n")
    econ_summary = {}
    for tk, key in tickers.items():
        if tk not in all_results:
            continue
        stk = market[key]
        fwd_ret = stk["Close"].pct_change().shift(-1)
        models = all_results[tk]["models"]
        best = max(
            (k for k in models if k.startswith(("LR", "RF"))),
            key=lambda k: (models[k]["auc"] if isinstance(models[k]["auc"], float)
                           and not np.isnan(models[k]["auc"]) else -1),
            default=None,
        )
        if best is None:
            continue
        oos = all_results[tk]["oos"][best].dropna()
        econ = directional_backtest(fwd_ret, oos, mode="long_flat")
        econ_summary[tk] = {"best_model": best, **econ}
        print(f"  {tk:6} best={best:18} "
              f"strat_sharpe={econ['strategy_sharpe']:>6} vs BH={econ['buyhold_sharpe']:>6} | "
              f"strat_ret={econ['strategy_total_return']:>8} vs BH={econ['buyhold_total_return']:>8}")

    # ── Save + verdict ───────────────────────────────────────────────
    def _safe(o):
        import math
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, pd.Series):
            return None
        return str(o)

    out = {
        "task": f"{HORIZON}-day stock direction (up/down) detection",
        "walkforward": wf.__dict__,
        "results": {tk: {"direction_summary": r["direction_summary"], "models": r["models"]}
                    for tk, r in all_results.items()},
        "economic_backtest": econ_summary,
    }
    with open(cfg.paths.output_dir / "direction_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, default=_safe)

    # Aggregate edge over majority for ML models
    edges = []
    for tk, r in all_results.items():
        for name, m in r["models"].items():
            if name.startswith(("LR", "RF")) and isinstance(m["edge_over_majority"], float) \
               and not np.isnan(m["edge_over_majority"]):
                edges.append(m["edge_over_majority"])
    mean_edge = float(np.mean(edges)) if edges else float("nan")

    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    print(f"  Mean OOS accuracy EDGE of ML models over 'always-up' : {mean_edge:+.4f}")
    print(f"  Interpretation: {'ML adds directional skill' if mean_edge > 0.01 else 'No reliable edge — consistent with weak-form EMH'}")
    print(f"\n  Saved: {cfg.paths.output_dir / 'direction_metrics.json'}")
    return out


if __name__ == "__main__":
    main()
