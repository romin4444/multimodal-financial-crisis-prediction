# Milestone 5 — Consolidated Honest Findings

**Author:** Romin Patel · MBAI 5600G (Group 13)
**Data:** Real S&P 500 + VIX via yfinance, 1990-01-02 → 2024-12-30 (8,814 trading days).
**Label:** forward 21-day window containing a peak-to-trough drawdown ≥ 10% (n_positive = 346, base rate = 3.93%).
**Evaluation:** purged + embargoed walk-forward, baseline-normalized, calibrated probabilities, stationary block-bootstrap significance (B = 2000, block = 21d).

This file is the **single source of truth** for Milestone 5. Every number traces to a committed
JSON in `outputs/`. Where an earlier synthetic-sentiment run produced a different number, the
real-FinBERT result here **supersedes** it.

---

## 1. Crisis detection — nested ablation (real FinBERT, n = 7,407 OOS)

Source: `outputs/milestone5_results.json`.

| Model | PR-AUC | ROC-AUC | Brier skill | ECE | lift@10% |
|---|---|---|---|---|---|
| BASELINE VIX-threshold | 0.167 | 0.746 | −7.27 | 0.492 | 3.81 |
| price | 0.147 | 0.749 | −3.35 | 0.315 | 3.84 |
| price + macro | 0.091 | 0.588 | −3.78 | 0.307 | 2.89 |
| price + macro + online regime | 0.079 | 0.622 | −3.50 | 0.260 | 2.41 |
| price + macro + online + sentiment | 0.080 | 0.623 | −3.53 | 0.260 | 2.47 |
| FINAL calibrated (price) | 0.042 | 0.461 | −0.92 | **0.074** | 0.30 |

**Reading:** the best ML model (price-only) does **not** beat the VIX baseline; adding macro,
regime, or sentiment *monotonically degrades* ranking. Calibration's job is probability quality,
not ranking — it cuts ECE from 0.315 → 0.074. (In the base `v3_run.py` harness the calibrated
price-only model reaches Brier skill **+0.015**, ECE **0.017** — see `outputs/v3_metrics.json`.)

### Significance (stationary block bootstrap, B = 2000)

| Test | ΔPR-AUC | 95% CI | p(Δ ≤ 0) | Significant @5%? |
|---|---|---|---|---|
| best ML (price) vs VIX baseline | −0.020 | [−0.077, +0.024] | 0.78 | **No** |
| sentiment marginal (vs price+macro+online) | +0.0017 | [−0.0003, +0.0055] | 0.056 | **No** |

The honest claim is **parity with VIX, not superiority**, with *trustworthy* probabilities
(ECE 0.07). Sentiment's marginal value is borderline and not robust: +0.0017 (p = 0.056) in this
FRED-snapshot run, −0.0005 (p = 0.67) in the CPI-macro run (`milestone5_results` CPI variant).

---

## 2. Hazard / survival model (alternative architecture)

Source: `outputs/hazard_metrics.json` (FRED-snapshot run). 33 onsets, train 1990–2010, test 2010–2024.

| Horizon | C-index | base rate | N-day Brier skill (raw) | calibration |
|---|---|---|---|---|
| 21d | **0.862** | 0.116 | **+0.077** | identity (do-no-harm guard kept raw) |
| 63d | 0.862 | 0.241 | −0.136 | identity |

The discrete-time hazard model **ranks** drawdown-onset risk well (C-index 0.86). With only 33
onset events, sigmoid-OOF calibration would overfit, so the v3.4 guard correctly reports raw
probabilities (positive Brier skill at 21d). Hazard ratios per 1 SD (strongest drivers):
`drawdown_63` HR ≈ 0.11, `fedfunds_chg_126` HR ≈ 0.19, `yield_slope` HR ≈ 0.42.

---

## 3. Financial Stress Index — the strongest, most robust result

Source: `outputs/metrics_summary.json`.

| Metric | Result | Target | Status |
|---|---|---|---|
| FSI vs Fed STLFSI (correlation) | **r ≈ 0.80** | r > 0.60 | ✅ |
| FSI vs NBER recessions (AUC) | **0.861** | AUC > 0.80 | ✅ |
| Best GARCH by BIC | EGARCH(1,1) (BIC 22,782) | asymmetric for equity | ✅ leverage effect |
| HMM states retained | 3 (Stable/Volatile/Crisis) | 2–4 standard | ✅ |

A daily 4-component FSI reconstructs the St. Louis Fed's weekly STLFSI using only public market
data. This result is independent of the multimodal thesis and is the project's headline
contribution.

---

## 4. Risk overlay (deployable risk-reduction result)

Source: `outputs/risk_overlay_results.json` (SPY + VIX, 1993–2026).

| Strategy | CAGR | Sharpe | Max-DD |
|---|---|---|---|
| Buy & hold | 10.8% | 0.553 | −55.2% |
| Risk overlay | 6.6% | 0.646 | **−37.1%** |

| Bootstrap test | mean | 95% CI | significant? |
|---|---|---|---|
| Max-DD reduction | +19.0pp | [+5.9, +36.2] | **Yes (p ≈ 0.0005)** |
| Sharpe edge | +0.088 | [−0.086, +0.273] | No |

The overlay is a **risk-reduction tool, not an alpha tool**: it significantly cuts drawdown; the
Sharpe edge CI straddles zero, which is the EMH-consistent outcome.

---

## 5. Stock-direction prediction (robustness check)

Source: `outputs/direction_metrics.json`. 5-day up/down, AAPL/JPM/XOM/GS/^GSPC. Mean accuracy edge
over "always-up" ≈ **−0.05** across tickers → no reliable edge, consistent with weak-form EMH. A
second honest null on a second task increases confidence that the harness, not the task, is the
source of truth.

---

## 6. Base-paper (Wang et al. 2025) network reproduction

Source: `outputs/milestone5_reconciled.json`.

| Feature set | PR-AUC |
|---|---|
| price | 0.073 |
| price + Wang network | 0.069 |
| price + Wang + macro | 0.063 |
| price + Wang + macro + sentiment | 0.062 |

Network test: ΔPR-AUC = **−0.111**, 95% CI [−0.232, −0.015]. The heteroskedasticity-network
features were implemented and tested under leakage-free evaluation; they **do not add ranking
signal** for the 21-day drawdown target. Reported as an honest, measured null — a stronger
contribution than a passive citation.

---

## 7. Sentiment lead-lag — SYNTHETIC result superseded by REAL FinBERT

This is the most important correction in Milestone 5.

| Window | Synthetic proxy (earlier) | **REAL FinBERT (canonical)** |
|---|---|---|
| Overall | leads, Granger significant | **contemporaneous (lag 0)** |
| GFC 2008 | leads 9d, r = 0.77 | **lag 0, r = 0.10** |
| COVID 2020 | leads 5d, r = 0.83 | **lag 0, r = 0.51** |
| Granger (fear → regime) | significant at all lags 1–10 | **significant at NO lag (p > 0.16)** |

The apparent "sentiment leads price" signal was an artifact of the synthetic proxy being a
function of VIX (VIX autocorrelating with itself). With **real news sentiment**, the leading
signal disappears: sentiment is contemporaneous and does not Granger-cause the price regime. This
overturns the Milestone 1–2 premise and is an honest, publishable negative result. **All
lead-lag / Granger claims from the synthetic run are superseded by this section.**

---

## 8. Milestone-4 reconciliation (for the report opening)

Milestone 4 reported XGBoost PR-AUC = 0.232 "beating" VIX (0.170). That came from the v2 fusion
pipeline, which predicted the HMM's own crisis state from volatility features — a circular target,
the same leakage class caught in the V2→V3 fix. Under the v3 leakage-free harness (exogenous
drawdown label, causal regime posteriors, purged walk-forward), the calibrated price-only model
scores PR-AUC ≈ 0.147–0.166, **statistically indistinguishable from the VIX baseline (≈ 0.167)**.
The honest finding is not "ML beats VIX" but "no model adds ranking signal beyond VIX for drawdown
prediction" — a valid null, with calibration now trustworthy (ECE 0.07–0.017 vs 0.31).

---

## 9. One-line summary

> Three genuine, data-backed results — the **FSI nowcast** (r ≈ 0.80 vs STLFSI, NBER AUC 0.86),
> the **hazard ranking** (C-index 0.86), and the **risk overlay** (−19pp max-DD, p ≈ 0.0005) —
> survive leakage-free out-of-sample scrutiny. The multimodal-drawdown thesis (macro, sentiment,
> network) does **not** add ranking signal beyond VIX, and the apparent sentiment lead-lag was a
> synthetic-data artifact that vanishes under real FinBERT. Honest nulls, honestly reported.
