"""
Main orchestration pipeline.
Coordinates all stages: data → features → models → analysis → outputs.
Replaces the monolithic main() in final.py with a modular, testable design.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import cfg
from src.logging_setup import get_logger, setup_logging

log = get_logger("pipeline")


@dataclass
class PipelineResult:
    feat: pd.DataFrame
    regime_df: pd.DataFrame
    daily_sent: pd.DataFrame
    fusion_df: pd.DataFrame
    trained: object
    evaluations: dict
    shap_results: dict
    lead_lag_results: dict
    val_df: pd.DataFrame
    best_garch: object
    best_hmm: object
    all_hmm: dict
    all_garch: list
    metrics: dict
    runtime_seconds: float


def run_pipeline(news_dir: Optional[Path] = None) -> PipelineResult:
    """
    Execute the complete multimodal financial crisis prediction pipeline.

    Args:
        news_dir: Directory containing news CSV files. Defaults to auto-detection.

    Returns:
        PipelineResult with all computed artifacts.
    """
    setup_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_dir=cfg.paths.log_dir,
    )
    _seed_everything()
    t0 = time.time()
    log.info("Pipeline starting", extra={"seed": cfg.seed})

    # ── 1. Data acquisition ──────────────────────────────────────────
    log.info("Stage 1/9: Data acquisition")
    from src.data.market import download_all_market
    from src.data.fred import download_fred, align_fred_to_trading_days
    from src.data.news import load_news

    market = download_all_market()
    fred_df = download_fred()
    news_df = load_news(news_dir)
    sp500, vix = market["sp500"], market["vix"]

    # ── 2. Feature engineering ───────────────────────────────────────
    log.info("Stage 2/9: Feature engineering")
    from src.features.engineering import engineer_features, validate_features

    feat = engineer_features(sp500, vix)
    validate_features(feat)
    trading_idx = feat.index
    fred_daily = align_fred_to_trading_days(fred_df, trading_idx)

    # ── 3. FSI — initial build (GARCH = 0 placeholder) ──────────────
    log.info("Stage 3/9: Financial Stress Index (initial)")
    from src.models.fsi import FSIBuilder

    fsi_builder = FSIBuilder()
    feat, fsi_comps = fsi_builder.build(feat, fred_daily)

    # ── 4. GARCH volatility modelling ────────────────────────────────
    log.info("Stage 4/9: ARMA-GARCH volatility modelling")
    from src.models.garch import select_garch

    best_garch, all_garch = select_garch(feat["log_ret"].dropna())
    cond_var = _align_garch_var(best_garch.cond_var, feat)

    # ── 5. FSI — update with GARCH ───────────────────────────────────
    log.info("Stage 5/9: FSI update with GARCH variance")
    feat = fsi_builder.update_with_garch(feat, cond_var)
    # Capture FSI validity now — pandas .join() drops .attrs downstream
    fsi_validity = dict(feat.attrs.get("fsi_validity", {}))

    # ── 6. HMM regime detection ──────────────────────────────────────
    log.info("Stage 6/9: HMM regime detection")
    from src.models.hmm import select_and_fit_hmm

    best_hmm, all_hmm = select_and_fit_hmm(feat)
    regime_df = best_hmm.predict_regime(feat)
    feat = feat.join(regime_df, how="left")

    # ── 7. FinBERT sentiment ─────────────────────────────────────────
    log.info("Stage 7/9: FinBERT sentiment pipeline")
    from src.models.sentiment import run_finbert, aggregate_sentiment, run_vader, merge_real_and_synthetic

    fb_scores = run_finbert(news_df)
    daily_sent = aggregate_sentiment(fb_scores, trading_idx)
    daily_sent = merge_real_and_synthetic(daily_sent, feat)
    vader_sent = run_vader(news_df, trading_idx)

    # ── 8. Lead-lag & Granger causality ─────────────────────────────
    log.info("Stage 8/9: Lead-lag cross-correlation & Granger causality")
    from src.analysis.lead_lag import run_all_lead_lag, run_granger

    ll_results = run_all_lead_lag(feat, daily_sent)
    gc_df = run_granger(feat["FSI"], daily_sent["fear_index"])

    # ── 9. Fusion model ──────────────────────────────────────────────
    log.info("Stage 9/9: Fusion model + SHAP + evaluation")
    from src.models.fusion import build_fusion_matrix, train_fusion
    from src.analysis.shap_explain import run_shap
    from src.analysis.benchmarks import benchmark_wang2025, compare_finbert_vader
    from src.evaluation.validation import validate_checklist

    fusion_df = build_fusion_matrix(regime_df, daily_sent, feat)
    trained, evaluations = train_fusion(fusion_df)
    shap_results, _ = run_shap(fusion_df, trained)

    wang_df = benchmark_wang2025(regime_df)
    fb_vs_vader = compare_finbert_vader(daily_sent, vader_sent, feat["FSI"])
    val_df = validate_checklist(regime_df, daily_sent, evaluations)

    # ── Visualisations ───────────────────────────────────────────────
    log.info("Generating visualisations")
    _generate_plots(feat, regime_df, daily_sent, ll_results, shap_results, all_hmm, all_garch, evaluations)

    # ── Integration CSV ──────────────────────────────────────────────
    integration = _build_integration_csv(feat, regime_df, daily_sent)
    integration.to_csv(cfg.paths.output_dir / "integration_master.csv")
    log.info("Integration CSV saved", extra={"rows": len(integration)})

    # ── Metrics summary ──────────────────────────────────────────────
    feat.attrs["fsi_validity"] = fsi_validity  # restore for downstream
    metrics = _build_metrics(best_garch, best_hmm, all_hmm, feat, ll_results, evaluations, fb_vs_vader)
    with open(cfg.paths.output_dir / "metrics_summary.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    elapsed = time.time() - t0
    _log_summary(elapsed, best_garch, best_hmm, all_hmm, feat, ll_results, evaluations, integration)

    return PipelineResult(
        feat=feat, regime_df=regime_df, daily_sent=daily_sent,
        fusion_df=fusion_df, trained=trained, evaluations=evaluations,
        shap_results=shap_results, lead_lag_results=ll_results,
        val_df=val_df, best_garch=best_garch, best_hmm=best_hmm,
        all_hmm=all_hmm, all_garch=all_garch, metrics=metrics,
        runtime_seconds=elapsed,
    )


# ─── Private helpers ──────────────────────────────────────────────────────────

def _seed_everything() -> None:
    import random
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)


def _align_garch_var(cond_var: pd.Series, feat: pd.DataFrame) -> pd.Series:
    if not isinstance(cond_var, pd.Series):
        cond_var = pd.Series(cond_var, index=feat.index[: len(cond_var)], name="garch_var")
    return cond_var.reindex(feat.index).ffill()


def _generate_plots(feat, regime_df, daily_sent, ll_results, shap_results, all_hmm, all_garch, evaluations) -> None:
    from src.visualization.plots import (
        plot_regime_timeline, plot_sentiment_fsi, plot_lead_lag, plot_shap,
        plot_hmm_selection, plot_garch, plot_fusion_eval, plot_research_comparison,
    )
    try:
        plot_regime_timeline(feat, regime_df)
        plot_sentiment_fsi(feat, daily_sent)
        plot_lead_lag(ll_results)
        plot_shap(shap_results)
        plot_hmm_selection(all_hmm)
        plot_garch(feat, all_garch)
        plot_fusion_eval(evaluations)
        plot_research_comparison()
    except Exception as exc:
        log.error("Visualisation error", extra={"error": str(exc)})


def _build_integration_csv(feat, regime_df, daily_sent) -> pd.DataFrame:
    # Include EVERY column the trained fusion models need (so the API can
    # serve predictions without zero-padding missing features)
    keep = [c for c in [
        "regime", "prob_stable", "prob_volatile", "prob_crisis",
        "FSI", "vol_21d", "vix", "drawdown_63",
    ] if c in feat.columns]
    df = feat[keep].copy()
    for col in ["fear_index", "fear_3d", "fear_7d", "panic_signal", "headline_count", "is_synthetic"]:
        if col in daily_sent.columns:
            df[col] = daily_sent[col].reindex(df.index)
    return df


def _build_metrics(best_garch, best_hmm, all_hmm, feat, ll_results, evaluations, fb_vs_vader) -> dict:
    best_f1 = {
        crisis: round(max(ev.f1 for ev in evals), 4)
        for crisis, evals in evaluations.items()
        if evals
    }
    return {
        "best_garch": best_garch.label,
        "best_garch_bic": round(best_garch.bic, 2),
        "hmm_n_retained": best_hmm.n_states,
        "hmm_bic_profile": {n: round(r.bic, 2) for n, r in all_hmm.items()},
        "fsi_validity": feat.attrs.get("fsi_validity", {}),
        "lead_lag": {
            k: {"peak_lag": v["peak_lag"], "peak_r": round(v["peak_r"], 4), "interp": v["interp"]}
            for k, v in ll_results.items()
        },
        "fusion_best_f1_by_crisis": best_f1,
        "fusion_f1_target": cfg.fusion.f1_target,
        "finbert_vs_vader": fb_vs_vader,
    }


def _log_summary(elapsed, best_garch, best_hmm, all_hmm, feat, ll_results, evaluations, integration) -> None:
    best_f1 = {
        c: round(max(ev.f1 for ev in evals), 4)
        for c, evals in evaluations.items() if evals
    }
    fv = feat.attrs.get("fsi_validity", {})
    log.info(
        "Pipeline complete",
        extra={
            "runtime_minutes": round(elapsed / 60, 2),
            "best_garch": best_garch.label,
            "hmm_n_retained": best_hmm.n_states,
            "fsi_stlfsi_r": fv.get("stlfsi_r"),
            "fsi_nber_auc": fv.get("nber_roc_auc"),
            "lead_lag_overall": ll_results["overall"]["interp"],
            "fusion_f1": best_f1,
            "integration_rows": len(integration),
        },
    )
