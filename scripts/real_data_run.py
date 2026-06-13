#!/usr/bin/env python3
"""
Real-data run — yfinance market data + cached FRED + synthetic sentiment proxy.
This is the FULL pipeline minus FinBERT (which needs ~500 MB HuggingFace download).
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> dict:
    print("=" * 70)
    print("  FCPS REAL-DATA RUN")
    print("  Real S&P500 + VIX (yfinance) + Real FRED (cached) + Synthetic NLP")
    print("=" * 70)

    t0 = time.time()

    from src.config import cfg
    from src.logging_setup import setup_logging
    setup_logging(level="INFO", fmt="text", log_dir=cfg.paths.log_dir)

    # ── Data ─────────────────────────────────────────────────────────
    print("\n[1/6] Loading market data (yfinance / cache)...")
    from src.data.market import download_all_market
    market = download_all_market()
    sp500, vix = market["sp500"], market["vix"]
    print(f"  S&P 500: {len(sp500):,} rows {sp500.index.min().date()} -> {sp500.index.max().date()}")

    print("\n[2/6] Loading FRED data...")
    from src.data.fred import download_fred, align_fred_to_trading_days
    fred = download_fred()
    print(f"  FRED columns: {list(fred.columns)}")
    print(f"  STLFSI non-null: {fred['stl_fsi'].notna().sum():,}")

    print("\n[3/6] Engineering features...")
    from src.features.engineering import engineer_features, validate_features
    feat = engineer_features(sp500, vix)
    validate_features(feat)
    fred_daily = align_fred_to_trading_days(fred, feat.index)

    print("\n[4/6] Building FSI and fitting GARCH...")
    from src.models.fsi import FSIBuilder
    fsi_builder = FSIBuilder()
    feat, fsi_comps = fsi_builder.build(feat, fred_daily)

    from src.models.garch import select_garch
    best_garch, all_garch = select_garch(feat["log_ret"].dropna())

    import pandas as pd
    cond_var = best_garch.cond_var
    if not isinstance(cond_var, pd.Series):
        cond_var = pd.Series(cond_var, index=feat["log_ret"].dropna().index[:len(cond_var)], name="garch_var")
    cond_var = cond_var.reindex(feat.index).ffill()

    feat = fsi_builder.update_with_garch(feat, cond_var)
    fsi_validity = dict(feat.attrs.get("fsi_validity", {}))
    print(f"  FSI vs STLFSI r: {fsi_validity.get('stlfsi_r')}")
    print(f"  FSI NBER ROC-AUC: {fsi_validity.get('nber_roc_auc')}")
    print(f"  FSI target (r>0.60) met: {fsi_validity.get('target_met')}")

    print("\n[5/6] Running models...")
    print("  → HMM regime detection (40 seeds)...")
    cfg.hmm.n_init_seeds = 40
    from src.models.hmm import select_and_fit_hmm
    best_hmm, all_hmm = select_and_fit_hmm(feat)
    regime_df = best_hmm.predict_regime(feat)
    feat = feat.join(regime_df, how="left")

    print("  → Synthetic sentiment proxy (FinBERT skipped)...")
    from src.models.sentiment import build_synthetic_sentiment
    daily_sent = build_synthetic_sentiment(feat.dropna(subset=["log_ret", "vix"]))

    print("  → Lead-lag analysis...")
    from src.analysis.lead_lag import run_all_lead_lag, run_granger
    ll_results = run_all_lead_lag(feat, daily_sent)
    gc_df = run_granger(feat["FSI"], daily_sent["fear_index"])

    print("  → Fusion model training...")
    from src.models.fusion import build_fusion_matrix, train_fusion
    fusion_df = build_fusion_matrix(regime_df, daily_sent, feat)
    trained, evaluations = train_fusion(fusion_df)

    print("  → SHAP explainability...")
    from src.analysis.shap_explain import run_shap
    shap_results, _ = run_shap(fusion_df, trained)

    print("  → Wang2025 baseline + validation checklist...")
    from src.analysis.benchmarks import benchmark_wang2025
    from src.evaluation.validation import validate_checklist
    wang_df = benchmark_wang2025(regime_df)
    val_df = validate_checklist(regime_df, daily_sent, evaluations)

    print("\n[6/6] Generating outputs...")
    from src.visualization.plots import (
        plot_regime_timeline, plot_sentiment_fsi, plot_lead_lag,
        plot_shap, plot_hmm_selection, plot_garch, plot_fusion_eval,
        plot_research_comparison,
    )
    plot_regime_timeline(feat, regime_df)
    plot_sentiment_fsi(feat, daily_sent)
    plot_lead_lag(ll_results)
    plot_shap(shap_results)
    plot_hmm_selection(all_hmm)
    plot_garch(feat, all_garch)
    plot_fusion_eval(evaluations)
    plot_research_comparison()

    # Include EVERY column the trained fusion models need (so the API can
    # serve predictions without zero-padding missing features)
    keep = [c for c in [
        "regime", "prob_stable", "prob_volatile", "prob_crisis",
        "FSI", "vol_21d", "vix", "drawdown_63",  # required by fusion features
    ] if c in feat.columns]
    integration = feat[keep].copy()
    for col in ["fear_index", "fear_3d", "fear_7d", "panic_signal", "headline_count", "is_synthetic"]:
        if col in daily_sent.columns:
            integration[col] = daily_sent[col].reindex(integration.index)
    integration.to_csv(cfg.paths.output_dir / "integration_master.csv")

    best_f1 = {c: round(max(ev.f1 for ev in evals), 4) for c, evals in evaluations.items() if evals}
    metrics = {
        "best_garch": best_garch.label,
        "best_garch_bic": round(best_garch.bic, 2),
        "best_garch_aic": round(best_garch.aic, 2),
        "best_garch_lb_p": best_garch.lb_p,
        "best_garch_arch_p": best_garch.arch_p,
        "all_garch_results": [g.to_dict() for g in all_garch],
        "hmm_n_retained": best_hmm.n_states,
        "hmm_bic_profile": {n: round(r.bic, 2) for n, r in all_hmm.items()},
        "hmm_ll_profile": {n: round(r.log_likelihood, 2) for n, r in all_hmm.items()},
        "fsi_validity": fsi_validity,
        "fsi_target_correlation": cfg.fsi.correlation_target,
        "lead_lag": {k: {"peak_lag": v["peak_lag"], "peak_r": round(v["peak_r"], 4), "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"], "interp": v["interp"]} for k, v in ll_results.items()},
        "granger_causality": gc_df.to_dict("records") if not gc_df.empty else [],
        "fusion_best_f1_by_crisis": best_f1,
        "fusion_f1_target": cfg.fusion.f1_target,
        "fusion_evaluations": {c: [{"model": ev.model_name, "f1": ev.f1, "precision": ev.precision, "recall": ev.recall, "auc": ev.roc_auc, "n": ev.n_samples, "pos": ev.n_positive, "threshold": ev.threshold} for ev in evals] for c, evals in evaluations.items()},
        "wang2025_benchmark": wang_df.to_dict("records"),
        "validation_checklist": val_df.to_dict("records"),
        "data_source": {
            "market": "yfinance (REAL S&P 500 + VIX 1990-2024)",
            "fred": "REAL FRED (cached fred_data.csv)",
            "sentiment": "Synthetic VIX-based proxy (FinBERT skipped)",
        },
    }
    from src.json_utils import safe_json_default

    with open(cfg.paths.output_dir / "metrics_summary.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=safe_json_default)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  REAL-DATA RUN COMPLETE — {elapsed/60:.1f} minutes")
    print("=" * 70)
    print(f"\n  GARCH        : {best_garch.label} | BIC={best_garch.bic:.2f}")
    print(f"  HMM          : n={best_hmm.n_states} retained")
    print(f"  HMM BIC      : {dict((n, round(r.bic)) for n, r in all_hmm.items())}")
    print(f"  FSI vs STLFSI: r={fsi_validity.get('stlfsi_r')} (target r>0.60: {fsi_validity.get('target_met')})")
    print(f"  FSI NBER AUC : {fsi_validity.get('nber_roc_auc')}")
    print(f"  Lead-lag     : {ll_results['overall']['interp']} r={ll_results['overall']['peak_r']:.4f}")
    print(f"  Fusion F1    : {best_f1}")
    print(f"\n  Outputs: {cfg.paths.output_dir}/")
    for f in sorted(cfg.paths.output_dir.glob("*.*")):
        if f.is_file():
            print(f"    {f.name}  ({f.stat().st_size/1024:.0f} KB)")

    return metrics


if __name__ == "__main__":
    main()
