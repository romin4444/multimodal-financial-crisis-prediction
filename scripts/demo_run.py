#!/usr/bin/env python3
"""
End-to-end demo run using cached / synthetic data.
Validates the full pipeline without needing yfinance, FRED API, or HuggingFace.

Use this for:
  - CI smoke tests
  - Local development without internet
  - Verifying the modular refactor works
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so log unicode chars don't crash
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(start: str = "1990-01-01", end: str = "2024-12-31", seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with realistic crisis-period dynamics."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, freq="B")
    n = len(idx)

    # Base drift
    log_ret = rng.normal(0.0003, 0.01, size=n)

    # Inject crisis-period volatility shocks (matches our config crisis windows)
    crisis_periods = [
        ("2008-09-01", "2009-03-31", 0.04),   # GFC
        ("2020-02-19", "2020-03-23", 0.06),   # COVID
        ("2022-01-01", "2022-10-31", 0.025),  # Inflation
    ]
    for cs, ce, sigma in crisis_periods:
        mask = (idx >= pd.Timestamp(cs)) & (idx <= pd.Timestamp(ce))
        log_ret[mask] += rng.normal(-0.002, sigma, size=mask.sum())

    price = 300 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(
        {
            "Open": price * (1 + rng.uniform(-0.002, 0.002, n)),
            "High": price * (1 + rng.uniform(0.001, 0.005, n)),
            "Low": price * (1 - rng.uniform(0.001, 0.005, n)),
            "Close": price,
            "Volume": rng.integers(1_000_000_000, 5_000_000_000, n).astype(float),
        },
        index=idx,
    )


def make_synthetic_vix(start: str = "1990-01-01", end: str = "2024-12-31", seed: int = 99) -> pd.DataFrame:
    """Synthetic VIX with crisis spikes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, freq="B")
    n = len(idx)
    base = 15 + 5 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = rng.normal(0, 2, n)
    vix = base + noise

    # Crisis spikes
    crisis_periods = [
        ("2008-09-01", "2009-03-31", 50),
        ("2020-02-19", "2020-03-23", 65),
        ("2022-01-01", "2022-10-31", 30),
    ]
    for cs, ce, peak in crisis_periods:
        mask = (idx >= pd.Timestamp(cs)) & (idx <= pd.Timestamp(ce))
        vix[mask] = vix[mask] + rng.uniform(peak * 0.5, peak, size=mask.sum())

    vix = np.clip(vix, 9, 80)
    return pd.DataFrame({"Close": vix}, index=idx)


def stage_synthetic_cache() -> None:
    """Pre-populate the cache directory with synthetic OHLCV so yfinance isn't called."""
    from src.config import cfg
    cache = cfg.paths.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    sp500 = make_synthetic_ohlcv()
    vix = make_synthetic_vix()

    sp500.to_csv(cache / "mkt_GSPC.csv")
    vix.to_csv(cache / "mkt_VIX.csv")

    # Individual stocks - similar dynamics with different seeds
    for i, ticker in enumerate(cfg.data.stock_tickers):
        stk = make_synthetic_ohlcv(seed=42 + i)
        safe = ticker.replace("^", "").replace("/", "-")
        stk.to_csv(cache / f"mkt_{safe}.csv")

    print(f"  [OK]Cached synthetic market data in {cache}")


def main() -> None:
    print("=" * 70)
    print("  FCPS DEMO RUN — synthetic data, no external APIs required")
    print("=" * 70)

    t0 = time.time()

    # ── Stage cache ──────────────────────────────────────────────────
    print("\n[1/6] Staging synthetic market data...")
    stage_synthetic_cache()

    # ── Import after cache is staged ────────────────────────────────
    from src.config import cfg
    from src.logging_setup import setup_logging

    setup_logging(level="INFO", fmt="text", log_dir=cfg.paths.log_dir)

    print("\n[2/6] Loading market data from cache...")
    from src.data.market import download_all_market
    market = download_all_market()
    sp500, vix = market["sp500"], market["vix"]
    print(f"  [OK]S&P 500: {len(sp500):,} rows | VIX: {len(vix):,} rows")

    print("\n[3/6] Engineering features...")
    from src.features.engineering import engineer_features, validate_features
    feat = engineer_features(sp500, vix)
    validate_features(feat)
    print(f"  [OK]Feature matrix: {feat.shape}")

    # Load cached FRED
    print("\n[4/6] Loading FRED data (from cache or skip)...")
    from src.data.fred import download_fred, align_fred_to_trading_days
    fred_df = download_fred()
    fred_daily = align_fred_to_trading_days(fred_df, feat.index)
    print(f"  [OK]FRED columns: {list(fred_daily.columns) if not fred_daily.empty else 'NONE (will use VIX proxy)'}")

    print("\n[5/6] Running models...")

    # FSI
    print("  → FSI (initial)...")
    from src.models.fsi import FSIBuilder
    fsi_builder = FSIBuilder()
    feat, fsi_comps = fsi_builder.build(feat, fred_daily)

    # GARCH
    print("  → GARCH...")
    from src.models.garch import select_garch
    best_garch, all_garch = select_garch(feat["log_ret"].dropna())

    cond_var = best_garch.cond_var
    if not isinstance(cond_var, pd.Series):
        cond_var = pd.Series(cond_var, index=feat["log_ret"].dropna().index[:len(cond_var)], name="garch_var")
    cond_var = cond_var.reindex(feat.index).ffill()

    # FSI with GARCH
    print("  → FSI (with GARCH)...")
    feat = fsi_builder.update_with_garch(feat, cond_var)
    fsi_validity = dict(feat.attrs.get("fsi_validity", {}))  # capture before join wipes attrs

    # HMM
    print("  → HMM regime detection...")
    from src.models.hmm import select_and_fit_hmm
    # Reduce seeds for demo speed
    cfg.hmm.n_init_seeds = 10
    best_hmm, all_hmm = select_and_fit_hmm(feat)
    regime_df = best_hmm.predict_regime(feat)
    feat = feat.join(regime_df, how="left")

    # Synthetic sentiment (skip FinBERT for speed)
    print("  → Sentiment (synthetic proxy — no FinBERT in demo)...")
    from src.models.sentiment import build_synthetic_sentiment
    daily_sent = build_synthetic_sentiment(feat.dropna(subset=["log_ret", "vix"]))

    # Lead-lag
    print("  → Lead-lag analysis...")
    from src.analysis.lead_lag import run_all_lead_lag, run_granger
    ll_results = run_all_lead_lag(feat, daily_sent)
    gc_df = run_granger(feat["FSI"], daily_sent["fear_index"])

    # Fusion
    print("  → Fusion model training...")
    from src.models.fusion import build_fusion_matrix, train_fusion
    fusion_df = build_fusion_matrix(regime_df, daily_sent, feat)
    trained, evaluations = train_fusion(fusion_df)

    # SHAP
    print("  → SHAP explainability...")
    from src.analysis.shap_explain import run_shap
    shap_results, _ = run_shap(fusion_df, trained)

    # Benchmarks
    print("  → Research benchmarks...")
    from src.analysis.benchmarks import benchmark_wang2025
    wang_df = benchmark_wang2025(regime_df)

    # Validation
    from src.evaluation.validation import validate_checklist
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

    # Integration CSV
    keep = [c for c in ["regime", "prob_stable", "prob_volatile", "prob_crisis", "FSI"] if c in feat.columns]
    integration = feat[keep].copy()
    for col in ["fear_index", "fear_3d", "fear_7d", "panic_signal", "headline_count", "is_synthetic"]:
        if col in daily_sent.columns:
            integration[col] = daily_sent[col].reindex(integration.index)
    integration.to_csv(cfg.paths.output_dir / "integration_master.csv")

    # Metrics summary
    best_f1 = {c: round(max(ev.f1 for ev in evals), 4) for c, evals in evaluations.items() if evals}
    metrics = {
        "best_garch": best_garch.label,
        "best_garch_bic": round(best_garch.bic, 2),
        "hmm_n_retained": best_hmm.n_states,
        "hmm_bic_profile": {n: round(r.bic, 2) for n, r in all_hmm.items()},
        "fsi_validity": fsi_validity,
        "lead_lag": {k: {"peak_lag": v["peak_lag"], "peak_r": round(v["peak_r"], 4), "interp": v["interp"]} for k, v in ll_results.items()},
        "fusion_best_f1_by_crisis": best_f1,
        "fusion_f1_target": cfg.fusion.f1_target,
        "data_source": "SYNTHETIC (demo run)",
    }
    with open(cfg.paths.output_dir / "metrics_summary.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  DEMO COMPLETE — {elapsed/60:.1f} minutes")
    print("=" * 70)
    print(f"\n  Best GARCH        : {best_garch.label} (BIC={best_garch.bic:.2f})")
    print(f"  HMM states        : n={best_hmm.n_states} retained")
    print(f"  HMM BIC profile   : {dict((n, round(r.bic, 0)) for n, r in all_hmm.items())}")
    print(f"  FSI validity      : NBER-AUC={fsi_validity.get('nber_roc_auc')} "
          f"STLFSI-r={fsi_validity.get('stlfsi_r')} "
          f"target_met={fsi_validity.get('target_met')}")
    print(f"  Lead-lag overall  : {ll_results['overall']['interp']} (r={ll_results['overall']['peak_r']:.4f})")
    print(f"  Fusion F1 by crisis: {best_f1}")
    print(f"\n  Output files in   : {cfg.paths.output_dir}/")
    for f in sorted(cfg.paths.output_dir.glob("*.*")):
        if f.is_file():
            print(f"    {f.name}  ({f.stat().st_size/1024:.0f} KB)")

    return metrics


if __name__ == "__main__":
    main()
