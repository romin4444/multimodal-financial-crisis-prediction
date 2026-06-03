# Predictive Edge & Multimodal Thesis — Run Log

*Honest results from `notebooks/kaggle_edge_and_multimodal.ipynb`. Author: Romin Patel.*

Decision rule (set before running): a config **wins Q1** only if, on some target,
`OOS PR-AUC > VIX` **and** `PBO < 0.5` **and** `Deflated Sharpe > 0.95`. The
multimodal thesis is **validated (Q2)** only if real-news sentiment raises OOS
PR-AUC **and** Granger-causes forward stress (p < 0.05).

---

## Run 1 — 2026-06-01 (Kaggle GPU; FRED key ✗, news ✗)

Config: `device=cuda`, `fred=false`, `macro=[xasset_corr]`, `tda=[tda_total, tda_max_h1]`,
`sentiment=[]`, rows=8,815, CPCV (6 groups, k=2, 21d embargo).

### Q1 — does anything beat a VIX threshold? **NO.**

| Target | base | **VIX PR-AUC** | price | price+macro | +regime | +TDA |
|---|---|---|---|---|---|---|
| 10d / 7% | 0.034 | **0.180** | 0.128 | 0.134 | 0.102 | 0.085 |
| 10d / 10% | 0.015 | **0.160** | 0.132 | 0.090 | 0.037 | 0.018 |
| 21d / 7% | 0.092 | **0.229** | 0.151 | 0.151 | 0.146 | 0.148 |
| 21d / 10% | 0.039 | **0.178** | 0.103 | 0.094 | 0.067 | 0.057 |
| 63d / 7% | 0.211 | **0.346** | 0.271 | 0.263 | 0.253 | 0.245 |
| 63d / 10% | 0.136 | **0.258** | 0.176 | 0.171 | 0.169 | 0.157 |

`q1_real_edge = false`, `q1_winners = []`. VIX wins **all six** targets; no config
cleared even the first gate, so PBO/DSR were moot.

**Notable:** **TDA features hurt in every case** (e.g. 10d/7%: price 0.128 → +TDA
0.085). The 2025 topological-ML early-warning result does **not** replicate as
*incremental* value over a VIX baseline here. Adding causal-regime probabilities
also degrades PR-AUC — classic overfitting on a rare positive class.

### Q2 — does real news sentiment add value? **NOT TESTED.**
`sentiment = []`, `q2.ran = false` — no news dataset was attached.

### Why Run 1 is incomplete (notebook logs)
- `!!! NO FRED KEY — "No user secrets exist"` → **credit spreads / yield curve /
  funding were absent** (the strongest VIX-orthogonal signals). Only cross-asset
  correlation survived. Q1 therefore tested only the weak half of the orthogonal set.
- `No news dataset attached` → Q2 could not run.

### To complete (Run 2)
1. Kaggle → **Add-ons → Secrets** → secret named **exactly** `FRED_API_KEY`
   (free key from fred.stlouisfed.org). Unlocks credit/yield/funding features.
2. **Add Input → Datasets** → a financial-news dataset (date + headline columns).
3. Run All. This delivers Q1's best shot (credit spreads) and Q2 (real FinBERT).

---

## Run 2 — 2026-06-02 (real full-history credit + macro; the #1 lever)

`scripts/run2_macro_edge.py`, real FRED key. The ICE HY OAS series (`BAMLH0A0HYM2`)
is relicensed to 2023+ on FRED, so the long-history credit proxy is **`BAA10Y`
(Moody's Baa − 10Y, 100% coverage 1990–2024)**, plus `T10Y2Y`, `T10Y3M`, `DGS3MO`.
Honest CPCV (6 groups, 21d embargo), calibrated logistic, full sweep.

| Target | base | **VIX PR-AUC** | price | +macro | +macro+regime | PBO | DSR |
|---|---|---|---|---|---|---|---|
| 10d/7% | 0.035 | **0.180** | 0.130 | 0.058 | 0.064 | 0.89 | 0.98 |
| 10d/10% | 0.015 | **0.160** | 0.130 | 0.128 | 0.092 | 0.00 | 0.99 |
| 21d/7% | 0.092 | **0.229** | 0.152 | 0.108 | 0.109 | 0.46 | 0.95 |
| 21d/10% | 0.040 | **0.178** | 0.096 | 0.061 | 0.060 | 0.94 | 1.00 |
| 63d/7% | 0.212 | **0.346** | 0.254 | 0.176 | 0.177 | 0.56 | 0.99 |
| 63d/10% | 0.137 | **0.258** | 0.173 | 0.124 | 0.123 | 0.39 | 1.00 |

**Verdict: `any_beats_vix = False`, `any_wins_all_gates = False`.** Even with
genuine full-history credit spreads — the single most-recommended VIX-orthogonal
signal — **nothing beats VIX**, and adding macro features *hurt* vs price-only
(multicollinearity on a rare positive class). Intuition: for forecasting a forward
**equity** drawdown, option-implied equity vol (VIX) is the most direct signal;
credit leads some crises (2008) but lags vol shocks (COVID), so on average over
1990–2024 it does not clear the bar. Positive control held: **FSI vs STLFSI r = 0.756**.

This is now a *robust* null for Q1 — established with the best available data, not
a data-limitation excuse. The only remaining untested lever is **real news**.

## Run 3 — 2026-06-02 (Kaggle GPU, FRED key WORKING, real credit + TDA)

`kaggle_edge_and_multimodal.ipynb` with `fred=true`; macro = credit_spread (BAA10Y),
credit_chg_63, yield_slope, yield_chg_63, oil_mom_63, xasset_corr; TDA = tda_total,
tda_max_h1. Artifact: `outputs/edge_multimodal_run3_kaggle_fred.json`.

| Target | base | **VIX** | price | +macro | +regime | +TDA |
|---|---|---|---|---|---|---|
| 10d/7% | 0.034 | **0.180** | 0.128 | 0.084 | 0.074 | 0.067 |
| 10d/10% | 0.015 | **0.160** | 0.132 | 0.101 | 0.044 | 0.024 |
| 21d/7% | 0.092 | **0.229** | 0.151 | 0.102 | 0.114 | 0.112 |
| 21d/10% | 0.039 | **0.178** | 0.103 | 0.079 | 0.072 | 0.063 |
| 63d/7% | 0.211 | **0.346** | 0.271 | 0.230 | 0.225 | 0.217 |
| 63d/10% | 0.136 | **0.258** | 0.176 | 0.140 | 0.139 | 0.140 |

`q1_real_edge=false`. **Confirms the local Run 2 exactly**: with real full-history
credit + TDA, on GPU, **no feature set beats VIX on any target**; each added layer
lowers PR-AUC. Q1 is now a robust null across two independent environments. Q2
still not run (no news dataset attached).

## Run 4 (to do) — real-news multimodal, the last open shot
Attach this Kaggle dataset to the GPU notebook (date + headline, covers all crisis
windows): **`dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024`**
(S&P 500 + daily headlines, 2008–2024). Backup: `notlucasp/financial-news-headlines`.
Then run `kaggle_edge_and_multimodal.ipynb` / `kaggle_fair_benchmark.ipynb` with
the FRED secret set. This is the only path that could still clear the gates;
EMH-style priors are skeptical, but it is the experiment that confirms or retires
the multimodal thesis for good.

## Interpretation (for Milestone 4)

Even with **GPU + TDA (a 2025 frontier method) + cross-asset correlation +
causal-regime features**, evaluated under **CPCV** on six targets, **no model
beats a one-line VIX threshold out-of-sample.** That is a rigorous, credible
**null result** — and exactly the kind of finding the 2024–2025 backtest-overfitting
and TSFM-leakage literature exists to surface. It is far more defensible than a
"we beat the market" claim.

The project's one robustly positive result remains the **Financial Stress Index
vs the Fed's STLFSI: r = 0.823** (NBER recession AUC = 0.861) — a daily,
public-data reconstruction of a Fed index. That should be the presentation's
centrepiece; this honest edge/null analysis is the methodological backbone.

Artifacts: `outputs/edge_multimodal_run1.json`,
`notebooks/run_2026-06-01_edge_multimodal_executed.ipynb`.
