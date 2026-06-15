# Final Multimodal Project Review — Responses (honest, data-backed)

> Every claim below is backed by a committed artifact in `outputs/` and reproducible from the
> scripts in `scripts/`. Where an earlier draft overstated results (network feature value,
> cross-asset topology generalization, sentiment lead-lag), this version reports what the
> leakage-free harness actually measured.

**1. Dataset scope (generalizability).**
The crisis model is trained and evaluated on the S&P 500 over 1990–2024 (8,814 trading days,
3.93% base rate). Cross-market generalization of the *crisis* model is **not yet complete** — it
remains S&P-500-only — and is scoped for Milestone 6 (the v3 harness is ready to run on
`^STOXX50E` / `^N225` / `^FTSE`). The *direction* task was run on five tickers
(AAPL/JPM/XOM/GS/^GSPC) and produced a consistent null (no edge over "always-up"), which is itself
evidence the harness — not the specific asset — drives the result. We do not claim cross-asset
topology generalization that we have not measured.

**2. Prediction objective.**
The target is the **onset of a major market crisis**, defined as a future 21-day window containing
a peak-to-trough drawdown ≥ 10%. This is an *exogenous* label (independent of the model's own
regime state), which removes the circular-target leakage present in the Milestone-4 v2 pipeline.

**3. Integration of the two pipelines (econometric + NLP).**
The model uses **early (feature) fusion**: price/volatility features, FRED macro stress features,
HMM regime posteriors, and FinBERT sentiment are concatenated into one temporal matrix, imputed,
and passed to a Random Forest (plus LR/GB) classifier. We also implemented heteroskedasticity-
network features (per Wang et al. 2025) and tested them in the same fusion. **Honest finding:** in
leakage-free out-of-sample evaluation, the macro, sentiment, and network components each *degrade*
ranking relative to price-only (e.g. price 0.147 → +macro 0.091 → +sentiment 0.080; network test
ΔPR-AUC −0.111). The multimodal architecture is fully implemented; the data simply does not reward
it for drawdown ranking on this window.

**4. Performance evaluation.**
The primary metric for this imbalanced task is **PR-AUC**, reported alongside ROC-AUC, Expected
Calibration Error (ECE), and Brier Skill Score (BSS). Significance is tested with a stationary
block bootstrap (Politis & Romano 1994; B = 2000, block = 21d) against a VIX-threshold baseline.
Results: best ML (price-only) PR-AUC 0.147 vs VIX 0.167, ΔPR-AUC −0.020, 95% CI [−0.077, +0.024],
**not significant** — i.e. parity with VIX, not superiority. Calibration reduces ECE from 0.31 to
0.07 (to 0.017 in the base harness), making the probabilities trustworthy even though ranking does
not exceed the baseline.

**5. What genuinely works (the defensible contributions).**
- **FSI nowcast:** a daily 4-component Financial Stress Index reconstructs the Fed's weekly STLFSI
  at r ≈ 0.80 and classifies NBER recessions at AUC 0.861.
- **Hazard model:** discrete-time survival framing ranks drawdown onsets at C-index 0.862 with
  positive raw Brier skill (+0.077) at the 21-day horizon.
- **Risk overlay:** reduces maximum drawdown from −55% to −37% (−19pp, bootstrap p ≈ 0.0005); the
  Sharpe-edge CI straddles zero (EMH-consistent), so we claim risk reduction, not alpha.

**6. Sentiment lead-lag — correction.**
An earlier synthetic-sentiment run suggested news sentiment *leads* the price regime by 5–9 days
with significant Granger causality. Under **real FinBERT**, this does not hold: sentiment is
contemporaneous (peak lag 0) in every crisis window and Granger causality is not significant at any
lag (p > 0.16). We report this reversal explicitly rather than carry the synthetic result forward.

**7. Base-paper reproduction (Wang et al. 2025).**
We reproduced the core (log returns, GARCH-by-BIC, HMM regimes) and implemented network-derived
features to test Wang's thesis quantitatively. Deviations: full per-window ARMA-GARCH + KNN
symbolization were not reproduced (window length `w` and `k` unspecified in the paper; per-window
fitting is computationally heavy; the paper reports no ROC/PR-AUC for direct numerical comparison).
The network features were tested and found not to add ranking signal (ΔPR-AUC −0.111). This is a
measured, leakage-free null — a reproducibility result, not an omission.
