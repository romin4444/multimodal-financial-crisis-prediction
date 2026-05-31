"""
All visualizations. Each function saves one figure and logs the output path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("visualization")
sns.set_theme(style="whitegrid")

C = cfg.visualization.palette
DPI = cfg.visualization.dpi
OUT = cfg.paths.output_dir

STATE_COLORS = {0: C.get("stable", "#2ECC71"), 1: C.get("volatile", "#F39C12"), 2: C.get("crisis", "#E74C3C")}
STATE_LABELS = {0: "Stable", 1: "Volatile", 2: "Crisis"}


def plot_regime_timeline(feat: pd.DataFrame, regime: pd.DataFrame) -> Path:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    idx = feat.index.intersection(regime.index)
    price = feat["close"].loc[idx]
    reg = regime["regime"].loc[idx]
    fsi = feat["FSI"].loc[idx]

    a1.plot(price.index, price, color="#2C3E50", lw=0.7, zorder=5)
    a1.set_yscale("log")
    a1.set_title("S&P 500 with HMM Regime Labels", fontsize=14, fontweight="bold")
    a1.set_ylabel("S&P 500 (log scale)")

    for state in [0, 1, 2]:
        m = reg == state
        starts = m.index[m & ~m.shift(1, fill_value=False)]
        ends = m.index[m & ~m.shift(-1, fill_value=False)]
        for s_, e_ in zip(starts, ends):
            a1.axvspan(s_, e_, color=STATE_COLORS[state], alpha=0.20, zorder=2)

    _shade_crises(a1)
    patches = [mpatches.Patch(color=STATE_COLORS[s], label=STATE_LABELS[s], alpha=0.6) for s in range(3)]
    a1.legend(handles=patches, loc="upper left")

    a2.fill_between(fsi.index, fsi.values, color=C.get("fsi", "#9B59B6"), alpha=0.55)
    a2.set_ylabel("FSI")
    a2.set_ylim(0, 1)
    a2.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.6)
    a2.set_xlabel("Date")
    _shade_crises(a2, alpha=0.08)

    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUT / "01_regime_timeline.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved", extra={"file": path.name})
    return path


def plot_sentiment_fsi(feat: pd.DataFrame, sent: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    idx = feat.index.intersection(sent.index)

    for ax in axes:
        _shade_crises(ax, alpha=0.09)

    vix = feat["vix"].loc[idx]
    axes[0].plot(vix.index, vix.values, color=C.get("crisis", "#E74C3C"), lw=0.8)
    axes[0].fill_between(vix.index, vix.values, alpha=0.25, color=C.get("crisis", "#E74C3C"))
    axes[0].axhline(30, color="gray", ls="--", lw=0.8, label="VIX = 30")
    axes[0].set_ylabel("VIX")
    axes[0].legend(fontsize=9)
    axes[0].set_title("CBOE VIX Fear Gauge", fontweight="bold")

    fi = sent["fear_index"].reindex(idx)
    axes[1].plot(fi.index, fi.values, color=C.get("sentiment", "#3498DB"), lw=0.8)
    axes[1].fill_between(fi.index, fi.values, alpha=0.25, color=C.get("sentiment", "#3498DB"))
    axes[1].axhline(cfg.finbert.panic_threshold, color="orange", ls="--", lw=0.9,
                    label=f"Panic threshold ({cfg.finbert.panic_threshold})")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fear Index")
    axes[1].legend(fontsize=9)
    axes[1].set_title("FinBERT Sentiment Fear Index", fontweight="bold")

    fsi = feat["FSI"].reindex(idx)
    axes[2].plot(fsi.index, fsi.values, color=C.get("fsi", "#9B59B6"), lw=0.8)
    axes[2].fill_between(fsi.index, fsi.values, alpha=0.30, color=C.get("fsi", "#9B59B6"))
    axes[2].axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.6)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("FSI [0–1]")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Financial Stress Index (Composite)", fontweight="bold")

    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUT / "02_sentiment_vs_fsi.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved", extra={"file": path.name})
    return path


def plot_lead_lag(results: Dict[str, dict]) -> Path:
    keys = list(results.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    max_lag = cfg.lead_lag.max_lag_days
    for ax, key in zip(axes, keys):
        r = results[key]
        lags, corrs = r["lags"], r["corrs"]
        colors = [C.get("sentiment", "#3498DB") if lag > 0 else C.get("fsi", "#9B59B6") for lag in lags]
        ax.bar(lags, corrs, color=colors, alpha=0.7, width=0.85)
        ax.axvline(r["peak_lag"], color="red", ls="--", lw=1.5, label=f"Peak = {r['peak_lag']}d")
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.text(0.05, 0.97,
                f"95% CI: [{r['ci_lo']:.0f}, {r['ci_hi']:.0f}]d\nr = {r['peak_r']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.set_title(key.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Lag (days)\n← Sent lags FSI  |  Sent leads FSI →", fontsize=9)
        ax.set_ylabel("Pearson r")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8)
        ax.set_xlim(-max_lag - 1, max_lag + 1)

    plt.suptitle("Cross-Correlation: FSI vs FinBERT Fear Index", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUT / "03_lead_lag.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved", extra={"file": path.name})
    return path


def plot_shap(shap_results: dict) -> Path:
    by_crisis = shap_results.get("by_crisis", {})
    if not by_crisis:
        log.warning("No SHAP by-crisis data — skipping plot")
        return None
    n = len(by_crisis)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 7))
    if n == 1:
        axes = [axes]
    pal = [C.get("crisis"), C.get("volatile"), C.get("stable"), C.get("sentiment"),
           C.get("fsi"), "#8E44AD", "#16A085", "#D35400"]

    for ax, (crisis, sv) in zip(axes, by_crisis.items()):
        top = sv.head(8)
        bars = ax.barh(range(len(top)), top.values, color=pal[:len(top)], alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title(f"{crisis.replace('_', ' ')}\nSHAP Attribution", fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        for bar, val in zip(bars, top.values):
            ax.text(val + 5e-4, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8)

    plt.suptitle("SHAP Feature Importance by Crisis (Random Forest TreeExplainer)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = OUT / "04_shap_by_crisis.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved", extra={"file": path.name})
    return path


def plot_hmm_selection(all_hmm_results: dict) -> Path:
    ns = sorted(all_hmm_results.keys())
    bics = [all_hmm_results[n].bic for n in ns]
    lls = [all_hmm_results[n].log_likelihood for n in ns]
    best_n = ns[int(np.argmin(bics))]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(ns, bics, "o-", color=C.get("crisis", "#E74C3C"), lw=2, ms=8)
    a1.axvline(best_n, color="green", ls="--", label=f"BIC-min n={best_n}")
    a1.axvline(cfg.hmm.canonical_n_states, color="blue", ls=":", label=f"Canonical n={cfg.hmm.canonical_n_states}")
    a1.set_xticks(ns)
    a1.set_xlabel("n_states")
    a1.set_ylabel("BIC (↓ = better)")
    a1.set_title("HMM Model Selection via BIC", fontweight="bold")
    a1.legend()

    a2.plot(ns, lls, "s-", color=C.get("fsi", "#9B59B6"), lw=2, ms=8)
    a2.set_xticks(ns)
    a2.set_xlabel("n_states")
    a2.set_ylabel("Log-Likelihood")
    a2.set_title("HMM Log-Likelihood by n_states", fontweight="bold")

    plt.tight_layout()
    path = OUT / "05_hmm_selection.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved", extra={"file": path.name})
    return path


def plot_garch(feat: pd.DataFrame, all_garch: list) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    for ax in axes:
        _shade_crises(ax, alpha=0.08)

    axes[0].plot(feat["log_ret"] * 100, color="#2C3E50", lw=0.4, alpha=0.75)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("Log Return (%)")
    axes[0].set_title("S&P 500 Log Returns", fontweight="bold")

    for g in all_garch:
        cv = g.cond_vol
        if hasattr(cv, "reindex"):
            cv = cv.reindex(feat.index).ffill()
        axes[1].plot(feat.index, cv.values if hasattr(cv, "values") else cv,
                     lw=0.7, alpha=0.85, label=g.label)
    axes[1].set_ylabel("Annualised Vol (σ)")
    axes[1].set_title("GARCH-Family Conditional Volatility", fontweight="bold")
    axes[1].legend(fontsize=9)

    axes[2].plot(feat["vix"], color=C.get("garch", "#E67E22"), lw=0.8)
    axes[2].axhline(30, color="red", ls="--", alpha=0.5, label="VIX=30")
    axes[2].set_ylabel("VIX")
    axes[2].set_xlabel("Date")
    axes[2].set_title("CBOE VIX Index", fontweight="bold")
    axes[2].legend(fontsize=9)

    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUT / "06_garch_all.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved", extra={"file": path.name})
    return path


def plot_fusion_eval(evaluations: dict) -> Path:
    rows = []
    for crisis, evals in evaluations.items():
        for ev in evals:
            rows.append({
                "Crisis": crisis.replace("_", "\n"),
                "Model": ev.model_name,
                "f1": ev.f1, "prec": ev.precision,
                "rec": ev.recall, "auc": ev.roc_auc,
            })
    if not rows:
        log.warning("No evaluation data — skipping fusion eval plot")
        return None

    df = pd.DataFrame(rows)
    metrics = [m for m in ["f1", "prec", "rec", "auc"] if m in df.columns]
    labels = {"f1": "F1", "prec": "Precision", "rec": "Recall", "auc": "ROC-AUC"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        pivot = df.pivot(index="Crisis", columns="Model", values=m)
        pivot.plot(kind="bar", ax=ax, width=0.65, colormap="Set2")
        ax.set_title(labels.get(m, m), fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.axhline(cfg.fusion.f1_target, color="red", ls="--", alpha=0.7,
                   label=f"Target ({cfg.fusion.f1_target})")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Fusion Model Evaluation by Crisis Period", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT / "07_fusion_eval.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved", extra={"file": path.name})
    return path


def plot_research_comparison() -> Path:
    data = {
        "Study": ["Hamilton (1989)", "Bollen et al. (2011)", "Riso & Vacca (2024)",
                  "Bussmann et al. (2020)", "Ardia et al. (2020)", "THIS PROJECT (Group 13)"],
        "Method": ["HMM", "Granger causality", "GARCH+NLP",
                   "XAI credit risk", "MS-GARCH", "HMM+GARCH+FinBERT+SHAP"],
        "Price signals": ["✅", "❌", "✅", "✅", "✅", "✅"],
        "Sentiment": ["❌", "✅", "✅", "❌", "❌", "✅"],
        "Explainability": ["❌", "❌", "❌", "✅", "❌", "✅"],
        "Explicit lead-lag": ["❌", "Partial", "❌", "❌", "❌", "✅"],
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis("off")
    cc = [["#ECF0F1"] * len(df.columns)] * len(df)
    cc[-1] = ["#D5F5E3"] * len(df.columns)
    t = ax.table(cellText=df.values, colLabels=df.columns,
                 cellLoc="center", loc="center", cellColours=cc)
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.15, 2.3)
    for j in range(len(df.columns)):
        t[(0, j)].set_facecolor("#2C3E50")
        t[(0, j)].set_text_props(color="white", fontweight="bold")
    t[(len(df), 0)].set_text_props(fontweight="bold", color="#1A5276")
    ax.set_title("Comparison with Prior Literature (M2 Section 2.2)",
                 fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()
    path = OUT / "08_research_comparison.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info("Saved", extra={"file": path.name})
    return path


# ─── Private ─────────────────────────────────────────────────────────────────

def _shade_crises(ax, alpha: float = 0.10) -> None:
    for _, (cs, ce) in cfg.crisis_window_tuples.items():
        ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), color="red", alpha=alpha, zorder=1)
