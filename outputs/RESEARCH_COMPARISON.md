# FCPS — Research & Institutional Benchmark Comparison

**Data:** Real S&P 500 + VIX (1990–2024, 8,814 trading days) via yfinance · Real FRED economic
data (STLFSI, credit spread, fed funds, etc.) · **Real FinBERT sentiment** on the attached news
corpus (the synthetic VIX-proxy used in early demo runs is superseded throughout this document).

> **Honesty note.** An earlier version of this file reported a synthetic-sentiment run in which
> news sentiment appeared to *lead* the price regime and Granger-cause it at all lags. That signal
> was an artifact of the synthetic proxy being a function of VIX. Under **real FinBERT**, the
> leading signal disappears (see §3, Bollen comparison). This document now reports only
> real-data-backed results.

---

## 1. Headline numbers vs published targets

| Metric | Our result | Target / benchmark | Source | Status |
|---|---|---|---|---|
| FSI vs STLFSI (Fed) | **r ≈ 0.80** | r > 0.60 | M2 §4.3; Kliesen et al. (2012) | ✅ PASS |
| FSI vs NBER recessions | **AUC = 0.861** | AUC > 0.80 | Hatzius et al. (2010) | ✅ PASS |
| Best GARCH (BIC) | **EGARCH(1,1)** | asymmetric wins for equity | Nelson (1991) | ✅ leverage effect |
| HMM canonical states | **n = 3** | 2–4 standard | Ang & Timmermann (2012) | ✅ |
| Crisis ranking (best ML) vs VIX | PR-AUC 0.147 vs 0.167 | beat VIX | M4 target | ❌ **honest null** (ΔPR-AUC −0.02, NS) |
| Sentiment marginal value | ΔPR-AUC +0.0017 (p = 0.056) | significant lift | M1 thesis | ❌ **not significant** |
| Hazard concordance (21d) | **C-index 0.862** | discriminate onsets | — | ✅ |
| Risk-overlay max-DD reduction | **−19pp (p ≈ 0.0005)** | reduce tail risk | — | ✅ significant |
| Wang network marginal value | ΔPR-AUC −0.111 | reproduce paper's edge | Wang et al. (2025) | ❌ **measured null** |

**What passes:** the FSI nowcast, GARCH/HMM model selection, the hazard ranking, and the risk
overlay. **What is an honest null:** beating VIX on drawdown ranking, the multimodal lift (macro /
sentiment / network), and the sentiment lead-lag. Both categories are reportable; the project's
stated goal is to measure the gap, not hide it.

---

## 2. Comparison to academic literature

### Hamilton (1989) — Markov-switching for business cycles
BIC profile across n ∈ {2,3,4} decreases monotonically; we fix n = 3 (Stable/Volatile/Crisis) per
Ang & Timmermann for clean economic interpretation. Our HMM flags the GFC regime ~37 trading days
before the September-2008 onset — earlier than Hamilton's quarterly-GNP two-state model. (This is
a **regime-detection** lead, distinct from out-of-sample drawdown *prediction*, which does not
beat VIX — the two must not be conflated.)

### Bollen, Mao, Zeng (2011) — sentiment Granger-causes the market
Bollen reported Twitter mood Granger-causing the DJIA at lags 2–6 days. **We did not replicate
this with real news sentiment.** Under real FinBERT, news sentiment vs the price regime is
**contemporaneous (peak lag = 0)** in every window (GFC, COVID, Inflation-2022) and Granger
causality is **not significant at any lag (p > 0.16)**. Our earlier synthetic-proxy run appeared
to replicate Bollen, but that was VIX autocorrelating with itself. The honest finding is the
opposite of a replication: at daily frequency on this corpus, news sentiment does not lead the
price regime.

### Wang et al. (2025) — heteroskedasticity-network early warning
We reproduced the core components (log returns, GARCH-family selection by BIC, HMM regime
detection) and additionally implemented network-derived features to test Wang's central claim
quantitatively. Result: the network features **do not add ranking signal** for the 21-day drawdown
target (ΔPR-AUC = −0.111, 95% CI [−0.232, −0.015]). Full per-window KNN symbolization was not
reproduced (window length `w` and `k` are unspecified in the paper; per-window GARCH over thousands
of windows is computationally heavy, and the paper reports no ROC/PR-AUC for direct comparison).
Testing the thesis under leakage-free evaluation and reporting a measured null is a stronger
contribution than a passive citation.

### Ardia, Bluteau, Boudt (2020) — asymmetric volatility
EGARCH(1,1) wins by BIC over GARCH(1,1) (22,782 vs 23,046), confirming the leverage-effect
dominance Ardia documents. Ljung-Box p = 0.037 (mild residual AR); ARCH-LM p = 0.481 (no remaining
heteroskedasticity).

### Hatzius et al. (2010) — financial conditions indexes
A good FCI/FSI should reach ROC-AUC ≥ 0.80 for NBER recession classification. Our FSI: **AUC =
0.861**, and r ≈ 0.80 vs the published STLFSI using only daily public inputs. This is the strongest
single result in the project.

---

## 3. Comparison to financial-institution products

| Institution | Product | Our equivalent | Agreement |
|---|---|---|---|
| St. Louis Fed | STLFSI4 (weekly) | Daily FSI (4 components) | **r ≈ 0.80** ✅ |
| CBOE | VIX | VIX is a primary input | used directly |
| NBER | recession dating | validation flag | **AUC 0.861** ✅ |
| Goldman Sachs / Chicago Fed | FCI / NFCI | daily FSI composite | similar signal, smaller input set |

**Takeaway:** our FSI is correlation-equivalent to the St. Louis Fed's published STLFSI using
public-domain inputs and open-source code — deployable for daily stress monitoring at near-zero
marginal cost. The crisis-*prediction* layer, by contrast, does not beat a simple VIX threshold,
and we report that plainly.

---

## 4. Where our results match published literature

| Finding | Our result | Literature |
|---|---|---|
| Asymmetric vol (EGARCH > GARCH) for equity | EGARCH BIC 22,782 < GARCH 23,046 | Nelson (1991); Engle & Ng (1993) |
| Weak-form efficiency on short-horizon direction | no edge over always-up (−0.05) | Fama (1970) |
| FCI reconstructs official stress indices | r ≈ 0.80 vs STLFSI, AUC 0.86 vs NBER | Hatzius et al. (2010) |

---

## 5. Honest scorecard

- ✅ **FSI nowcast** — r ≈ 0.80 vs STLFSI, NBER AUC 0.86 (strongest result)
- ✅ **Hazard ranking** — C-index 0.862, raw Brier skill +0.077 (21d)
- ✅ **Risk overlay** — max-DD −19pp, p ≈ 0.0005 (significant); Sharpe edge not significant
- ✅ **Calibration** — ECE 0.31 → 0.07–0.017; reliability diagram in `outputs/reliability_diagram.png`
- ❌ **Beat VIX on drawdown ranking** — null (ΔPR-AUC −0.02, NS)
- ❌ **Multimodal lift** (macro / sentiment / network) — null; each degrades PR-AUC
- ❌ **Sentiment leads price** — synthetic artifact; real FinBERT is contemporaneous, Granger NS

All figures referenced here are produced by `scripts/real_data_run.py`, `scripts/v3_advanced_run.py`,
`scripts/hazard_run.py`, and `scripts/risk_overlay_run.py`, and are reproducible on the cached
1990–2024 window.
