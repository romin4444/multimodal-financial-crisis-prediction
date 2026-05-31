# FCPS — Full Project Review, Honest Problem Diagnosis & Futuristic Roadmap

*Multimodal Financial Crisis Prediction — MBAI 5600G, Group 13*
*Review date: 2026-05-31*

---

## PART A — Where are we? (Honest progress assessment)

### What is genuinely DONE and works

| Area | Status | Evidence |
|---|---|---|
| Modular architecture | ✅ Complete | `src/` with data / features / models / analysis / evaluation / visualization / api |
| Configuration management | ✅ Complete | `config.yaml` + Pydantic validation, env-var overrides |
| Structured logging | ✅ Complete | JSON logs with rotation (`src/logging_setup.py`) |
| Reproducible pipeline | ✅ Runs end-to-end | `scripts/real_data_run.py` — 4.6 min on real 1990–2024 data |
| REST API | ✅ Serves predictions | FastAPI, Pydantic schemas, tested |
| Test suite | ✅ Green in CI | `tests/` — unit + integration, incl. look-ahead/causality proofs (live count: README CI badge) |
| FSI vs Fed STLFSI | ✅ **r = 0.823** | This is real and strong (see caveat below) |
| **Honest OOS backtest** | ✅ **NEW (v3)** | `scripts/v3_run.py`, `outputs/v3_metrics.json` |

### Progress on a "0 → financial-institution-grade" scale

```
Research notebook ──────────────────────────────────────────► Institution-grade
   v1 (final.py)         v2 (modular)        v3 (honest eval)        Production
        │                     │                     │                    │
        ●─────────────────────●─────────────────────●····················○
      DONE                  DONE                  DONE (today)        ROADMAP
   2000-line              clean code,          leakage exposed     calibration,
   monolith               API, tests,          + walk-forward      live data,
                          BUT leaky eval       + baselines         monitoring,
                                                                   hazard models
```

**We are ~60% of the way to institution-grade.** The engineering scaffold is there. What was missing — and is the entire subject of this review — is that the **modeling and evaluation methodology was not yet honest**, so the headline "results" were not trustworthy. v3 fixes the *evaluation* and exposes the truth. The *modeling* improvements to actually beat the baseline are the forward roadmap (Part C).

---

## PART B — The Problems (code AND approach), now empirically proven

We did not just *assert* the v2 approach was flawed — we built v3 to **measure** it. Here is the proof.

### B.1 — The headline finding

| Metric | v2 claim | v3 honest result |
|---|---|---|
| Crisis-prediction skill | **F1 = 0.99** (in-sample) | **PR-AUC = 0.10** (walk-forward OOS) |
| Does the ML beat a 1-line VIX rule? | (never tested) | **No** — VIX PR-AUC 0.17 > full model 0.10 |
| Economic value of de-risking | (never tested) | **Negative** — total return 10.5 → 2.1 |

The v2 F1 of 0.99 was **almost entirely leakage**. Here is exactly why.

### B.2 — Problem 1: Circular target (the big one)

**Code**: `src/models/fusion.py:141-143`
```python
crisis_now = (regime["regime"] == 2).astype(int)          # HMM's own Crisis state
f["target"] = crisis_now.shift(-horizon)                  # predict it 5 days ahead
# ...while FEATURE_COLS includes: prob_crisis, FSI, vol_21d, vix, drawdown_63
```
The model predicts the HMM's *own* high-volatility cluster, using high-volatility features as input. Because volatility clusters (a 40-year-old stylized fact — Mandelbrot 1963), "high vol now → high vol in 5 days" is trivially true. **F1 = 0.99 measures volatility persistence, not crisis foresight.**

**Fix (v3)**: `src/v3/labeling.py` — target is the **exogenous forward 21-day drawdown ≤ −10%**, a tradable outcome independent of any model. Base rate: 3.9% of days. *Now the problem is genuinely hard.*

### B.3 — Problem 2: Look-ahead in the regime labels

**Code**: `src/models/hmm.py:195` → `model.predict_proba(X)`
`predict_proba` is the **forward-backward (smoothed)** posterior P(state_t | x_1…x_**T**). The backward pass uses the **entire** series, including the future. So v2's "regime at time t" knows what happens after t. The celebrated *"detected GFC 37 days early"* is partly an artifact — July 2008 is labeled Crisis using knowledge of the September crash.

**Fix (v3)**: `src/v3/causal_regime.py` — `filtered_state_proba()` implements the **forward-only** filter P(state_t | x_1…x_t). Unit test `test_filtered_is_truly_causal` proves the value at t is invariant to any data after t.

### B.4 — Problem 3: Non-deployable "event-based holdout"

**Code**: `src/models/fusion.py:230` — trains on all non-crisis days, tests on the three *known* crisis windows. You cannot know future crisis dates in advance, so this split **peeks at the answer**. Plus the tiny windows (COVID = 24 days, all positive) make F1 meaningless and ROC-AUC undefined (we literally got `NaN`).

**Fix (v3)**: `src/v3/walkforward.py` — expanding-window walk-forward with a **21-day embargo** (purging, López de Prado 2018) between train and test so the forward-label window cannot overlap training. 119 honest out-of-sample folds.

### B.5 — Problem 4: No baseline, wrong metrics

**Code**: v2 reported only F1 at a tuned threshold. No "could the simplest thing beat it?" check.

**Fix (v3)**: `src/v3/baselines.py` + `src/v3/metrics.py` — base-rate, VIX-threshold, and persistence baselines, scored with PR-AUC, Brier Skill Score, calibration error (ECE), lift, and a real economic backtest. **The humbling result: VIX alone beats the ML stack.**

### B.6 — Problem 5: Sentiment is VIX in disguise

When real news is absent, `build_synthetic_sentiment` sets `fear_index = f(VIX)`. But FSI also contains VIX (30%), and the HMM consumes FSI. So "sentiment leads FSI" and the Granger result are partly **VIX correlating with itself**. The v3 ablation (price-only vs +regime vs +sentiment) shows adding these features **lowers** PR-AUC (0.157 → 0.104) — they contribute noise and multicollinearity, not signal, on this synthetic-sentiment configuration.

### B.7 — Residual code issues (lower priority)

- GARCH is fit in-sample (acceptable as a feature generator, but should be walk-forward refit for a true backtest).
- FSI weights are fixed by hand, never optimized or stability-tested.
- No probability calibration anywhere (all models output distorted probabilities — every Brier Skill Score is negative).
- Single market, single asset class; fixed hard-coded crisis windows.

---

## PART C — The Futuristic Approach (what v3 starts, and what's next)

### C.1 — Already delivered in this pass (v3, runnable today)

| Module | What it fixes | Test |
|---|---|---|
| `src/v3/labeling.py` | Exogenous forward-drawdown target → kills circularity | `test_v3.py::TestLabeling` |
| `src/v3/causal_regime.py` | Forward-only filtered HMM posteriors → kills look-ahead | `TestCausalRegime` (3 causality proofs) |
| `src/v3/baselines.py` | VIX / persistence / base-rate benchmarks | `TestBaselines` |
| `src/v3/metrics.py` | PR-AUC, Brier skill, ECE, lift, economic backtest | `TestMetrics` |
| `src/v3/walkforward.py` | Purged, embargoed expanding-window OOS | `TestWalkForward::test_no_lookahead_in_oos` |
| `scripts/v3_run.py` | End-to-end honest backtest on real data | — |

**Run it:** `python scripts/v3_run.py` → `outputs/v3_metrics.json`

### C.2 — The forward roadmap (to actually beat the VIX baseline)

The honest v3 result reframes the goal. The question is no longer *"can we get F1 = 0.99?"* (we can, trivially, by cheating). It is: **"can any model add real, calibrated, economically-positive skill over a VIX threshold, out-of-sample?"** That is the genuine research frontier. Concrete next steps, in priority order:

**1. Probability calibration (quick win).**
Wrap every model in isotonic / Platt calibration fit on a *validation fold inside the walk-forward train window*. Target: Brier Skill Score > 0 and ECE < 0.03. Without this, no probability is trustworthy for risk sizing.

**2. Prove sentiment's marginal value with REAL news (not the VIX proxy).**
Run the full FinBERT pipeline (already implemented in `src/models/sentiment.py:run_finbert`) on a real headline corpus, then repeat the v3 ablation. Only if `+sentiment` beats `price-only` OOS does the multimodal thesis hold. This is the single most important experiment to validate the project's core claim.

**3. Reframe as a hazard / survival problem.**
Crisis onset is time-to-event. A discrete-time hazard model (or `lifelines`/`scikit-survival`) gives a properly-censored "probability of a ≥10% drawdown within N days" with calibrated uncertainty — the exact object a risk officer wants. Replaces the awkward binary-classification framing.

**4. Walk-forward everything (GARCH + HMM + FSI scaler).**
Refit GARCH and the HMM inside each fold; compute FSI scaler per fold. Removes the last residual in-sample contamination so the backtest is fully deployable.

**5. Richer, VIX-orthogonal features.**
Credit spreads (already in FRED), yield-curve slope, cross-asset correlation spikes, options skew, market breadth, funding stress (TED/SOFR-OIS). The goal is signal *not already in VIX* — that is the only way to beat the VIX baseline.

**6. Modern sequence models (carefully).**
Once a calibrated, leakage-free, baseline-beating pipeline exists, test Temporal Fusion Transformers / TCNs / state-space models (Mamba) for multi-horizon hazard forecasting. Only after — not before — the methodology is sound. Deep models amplify leakage if the harness is wrong.

**7. Drift monitoring & MLOps.**
Production: alert when rolling 90-day STLFSI agreement < 0.6; auto-retrain quarterly; log every prediction with feature snapshot for audit; conformal prediction intervals on every API response.

**8. Multi-market generalization.**
Apply the leakage-free harness to other indices / asset classes (the Vietnam appendix is a start). A crisis detector that only works on the S&P 500 in-sample is not a crisis detector.

### C.3 — Definition of done (institution-grade)

A deployable model must clear **all** of:
- [ ] OOS PR-AUC **>** VIX-threshold baseline (currently 0.17), walk-forward.
- [ ] Brier Skill Score **> 0** and ECE **< 0.03** (calibrated).
- [ ] Economic backtest: higher Sharpe **and** lower max-drawdown than buy-and-hold, net of costs.
- [ ] `+sentiment` ablation positive with **real** news (not VIX proxy).
- [ ] Stable across ≥ 2 markets and ≥ 2 disjoint time periods.

v3 gives us the **honest measuring stick** to evaluate every future change against these criteria.

---

## PART D — What was built in this pass

```
NEW in v3:
  src/v3/labeling.py          exogenous forward-drawdown crisis label
  src/v3/causal_regime.py     forward-only (filtered) HMM posteriors
  src/v3/baselines.py         base-rate / VIX / persistence benchmarks
  src/v3/metrics.py           PR-AUC, Brier skill, ECE, lift, economic backtest
  src/v3/walkforward.py       purged, embargoed expanding-window backtest
  scripts/v3_run.py           runnable honest backtest on real data
  tests/test_v3.py            12 tests incl. 3 look-ahead/causality proofs
  docs/PROJECT_REVIEW_AND_ROADMAP.md   this document
  outputs/v3_metrics.json     the honest numbers

Test suite: green in CI — unit + integration (v2 + v3 + advanced + API/FRED integration). The live count is the README CI badge.
```

### Bottom line

The v2 system is **well-engineered but was dishonestly evaluated**; its F1 = 0.99 was leakage. The one result that survives scrutiny is the **FSI ≈ Fed STLFSI correlation (r = 0.823)** — that is real and valuable. Everything else needs the v3 harness to be re-measured.

v3's contribution is **intellectual honesty made executable**: an exogenous target, causal regimes, purged walk-forward, real baselines, and proper metrics. It shows the true skill is modest (PR-AUC ~0.10–0.17, currently *behind* a VIX rule) — which is exactly the starting line every serious quant crisis-prediction effort actually begins from. The roadmap in Part C is how we cross it.
