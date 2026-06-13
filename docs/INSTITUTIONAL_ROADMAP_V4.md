# FCPS — Deep Analysis & Institution-Grade Roadmap (v4)

*Independent review of `multimodal-financial-crisis-prediction` — what's genuinely
strong, what's quietly broken, and the exact path to a contribution that beats the
original Milestone paper and stands next to institutional research.*

Reviewer pass date: 2026-06-13 · Scope: full `src/`, `src/v3/`, `scripts/`, `docs/`, `outputs/`.

---

## 0. The one-paragraph verdict

You have already done the hard, rare thing: you exposed your own leakage, rebuilt
the evaluation honestly (exogenous labels, causal filtered regimes, purged/embargoed
walk-forward, CPCV, PBO, Deflated Sharpe), and reported a **robust null** — no model
beats a one-line VIX threshold for forward equity-drawdown *ranking*. That null is
more credible than 90% of "we predict crises" papers and is itself a publishable
contribution to the backtest-overfitting literature. **The way to go "institution-level"
from here is not to manufacture alpha — it is to (1) fix two latent issues that are
suppressing your single best result, (2) harden the data layer to point-in-time so the
positive results survive a quant reviewer, and (3) generalize the two deployable signals
(FSI nowcast + drawdown hazard) across markets.** Do that and the paper writes itself —
and it is *true*, which the original was not.

---

## 1. Asset inventory — what actually survives scrutiny

Three results survive. Everything institution-grade should be built on these, not on
trying to resurrect the multimodal-drawdown thesis (which you have correctly retired).

| # | Result | Current number | Status | Why it matters |
|---|---|---|---|---|
| **A** | Daily FSI reconstructs Fed STLFSI | r ≈ 0.76–0.82; NBER AUC ≈ 0.86 | Real, strong | A daily, public-data nowcast of a *weekly* Fed index from 4 inputs. This is the crown jewel. |
| **B** | Drawdown-onset **hazard** model | C-index **0.84** OOS | Real, *under-reported* | Best discrimination in the whole project. Currently hobbled by a calibration bug (§2.1). |
| **C** | Stress-scaled **risk overlay** | Max-DD −55% → −37%, 95% CI [+5.9pp, +36.2pp] | Real, significant | The deployable risk-management product. Drawdown CI clears zero; Sharpe CI honestly doesn't. |

The null (VIX is hard to beat on *ranking*) is the methodological backbone. A, B, C are
the *contributions*. The reframe in §4 makes them a paper.

---

## 2. Code-level findings (specific, with file references)

### 2.1 🔴 HIGH VALUE / LOW EFFORT — the hazard model is fighting the same bug you already fixed once

**File:** `src/v3/hazard.py:124`
```python
LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0)
```

This is the **exact** re-weighting you diagnosed and removed in v3.1 for the fusion
classifier (your README explains it: on a ~4% base rate, `balanced` inflates predicted
probabilities ~20×, destroying Brier skill and ECE while leaving ranking intact). It is
still live in the hazard model — and the symptom matches perfectly:

> hazard C-index = **0.84** (ranking: excellent) but N-day Brier skill = **−2.07** (probabilities: unusable).

That is the *signature* of `class_weight="balanced"`, not of the `1−(1−h)^N` approximation
you blamed in `docs/ADVANCED_RESULTS.md`. **Action:**
1. Drop `class_weight="balanced"` so the hazard sees the true base rate.
2. Wrap the **cumulative-incidence** output (not the per-step hazard) in isotonic
   calibration fit on an internal validation slice — exactly the pattern in
   `src/v3/calibration.py`.

**Expected outcome:** Brier skill flips from −2.07 toward ≥ 0 while C-index ≈ 0.84 holds.
That single change converts your strongest result from "ranks risk well but probabilities
are decorative" into **"a calibrated, deployable P(≥10% drawdown within N days)"** — which
is *precisely* the object a risk officer buys. This is the highest ROI change in the repo.

### 2.2 🔴 The `1−(1−h)^N` cumulative incidence over-states absolute risk

**File:** `src/v3/hazard.py` (`HazardFit.cumulative_incidence`)

Treating today's per-day hazard as constant across the whole horizon compounds an
already-inflated hazard. Two clean replacements, in order of rigor:
- **Quick:** keep the per-step hazard but calibrate the *N-day* incidence directly (§2.1).
- **Proper:** model the N-day outcome with a horizon-indexed pooled-logistic (one model,
  `horizon` as a covariate, person-period expansion) so the incidence is learned, not
  compounded. This is the standard discrete-time survival construction and gives you
  *time-dependent* risk curves for free.

### 2.3 🟠 Point-in-time / vintage-data leakage in the FRED layer (the institutional gotcha)

**File:** `src/data/fred.py:121` — the request sends `observation_start` only. There is no
`realtime_start` / `realtime_end`, so you pull **today's revised** FRED values, including
revised STLFSI4 and revised credit spreads.

Why a quant reviewer will flag this immediately: STLFSI4 and macro series are **revised**.
Validating "my daily FSI reconstructs STLFSI at r = 0.82" against the *revised* STLFSI,
using *revised* inputs, is not the real-time signal you'd have had on the day. Some of the
correlation is hindsight. **Action:** pull **vintage** data via ALFRED
(`realtime_start`/`realtime_end` on the FRED API, or `fredapi.get_series_first_release` /
`get_series_as_of_date`). Re-run the FSI vs STLFSI comparison **as-of each date**.

This cuts both ways and that is the point: if r holds at ~0.7+ on *vintage* data, your
crown-jewel result becomes **bulletproof** and genuinely institution-grade. If it drops,
you've found and reported a real effect — still a win, and still honest. Either outcome is
a stronger paper than the current one. This is the single change that most separates
"good student project" from "a quant desk would actually trust this."

### 2.4 🟠 Survival evaluation is missing the metrics institutions expect

The hazard model reports C-index + a Brier number. A survival/early-warning model at
institution grade is expected to report:
- **Time-dependent AUC** (AUC(t) over the forecast horizon, not a single pooled number).
- **Integrated Brier Score (IBS)** over the horizon — the standard scalar for calibrated
  survival quality.
- **Competing risks** awareness: "recovery" vs "deeper drawdown" are different exits;
  a cause-specific or Fine–Gray framing is the honest object here.

`scikit-survival` gives you `cumulative_dynamic_auc` and `integrated_brier_score` directly
and slots into your existing walk-forward masks.

### 2.5 🟡 Economic backtest realism (overlay → product)

`src/v3/metrics.py:economic_backtest` is clean and leakage-free (expanding-quantile
threshold, shifted by 1 — good). To make the overlay a *product* claim:
- **Regime-conditional reporting.** The overlay should add the most value *in* high-stress
  regimes. Report Sharpe/DD **conditional on the stress signal's top tercile** — that's
  where a risk desk cares, and where your honest "no average alpha, but real tail
  protection" story is strongest.
- **Cost & capacity sensitivity.** You already have 2 bps round-trip; add a small grid
  (2/5/10 bps) and turnover/capacity so the drawdown-reduction CI is shown to be
  cost-robust, not a low-cost artifact.
- **Turnover & exposure path** as first-class outputs (a risk committee asks "how often
  does this de-risk, and for how long?").

### 2.6 🟡 Minor / hygiene
- `forward_max_drawdown` and `_realized_within` are O(n·horizon) Python loops. Fine for
  research, but vectorize (rolling-min / suffix-max) before you scale to many markets ×
  many horizons — you'll be running this hundreds of times in §4.
- Two different `metrics_summary.json` provenance lines (`SYNTHETIC (demo run)`) sit next
  to the real `RESEARCH_COMPARISON.md` numbers. Tag every artifact with `data_source` +
  `git_sha` so no reader ever conflates the synthetic demo with the real run.
- `fusion.py` still uses `class_weight="balanced"` (lines 203/210) — acceptable since v2 is
  quarantined and you flag its F1 as circular, but add a one-line code comment pointing to
  the v3.1 note so nobody "fixes" it by copying it forward.

---

## 3. What NOT to do (and why it would make the project worse)

- ❌ **Don't add a Temporal Fusion Transformer / LSTM to "finally beat VIX" on drawdown
  ranking.** You have a 3.9% positive rate and a robust null established across two
  environments under CPCV. A deep model will overfit and your own PBO will catch it. Your
  roadmap already says "deep models amplify leakage if the harness is wrong" — that
  instinct is correct. Sequence models belong on the *hazard* (§4.3), where signal exists.
- ❌ **Don't reopen the multimodal-drawdown thesis as stated.** You retired it honestly with
  real FinBERT (Δ PR-AUC = −0.018, Granger p = 0.14). Reopening the *same* question looks
  like p-hacking. Instead, ask a *different*, defensible multimodal question (§4.2).
- ❌ **Don't chase a positive Sharpe edge for the overlay.** The Sharpe CI straddles zero
  exactly as weak-form EMH predicts; claiming otherwise would re-introduce the dishonesty
  you removed. The overlay's product is **risk reduction**, and that CI clears zero. Sell
  that.

---

## 4. The institution-grade roadmap (v4) — three tracks

Framing principle: **rigor + the deployable positives, generalized**, beats "we beat the
market." Below, each track has a falsifiable success bar.

### Track 1 — Make the crown jewel bulletproof (FSI nowcast) ⭐ do this first

**Goal:** turn "r ≈ 0.82 on revised data" into "r ≈ 0.7+ on **point-in-time** data,
out-of-sample, across markets."

1. Vintage FRED via ALFRED (§2.3); re-run FSI vs STLFSI **as-of each date**.
2. **Optimize the FSI weights honestly** — they're hand-set today. Fit the 4-component
   weights on a rolling expanding window (constrained non-negative, sum-to-one) and report
   stability of the weights over time. A *stable* learned weighting that matches the Fed is
   a much stronger claim than fixed weights.
3. **Nowcasting framing.** Position it explicitly as "a **daily** nowcast of a **weekly**
   Fed index" and evaluate the lead time: on the days between STLFSI releases, does your
   daily FSI anticipate the next print? That's a genuine, novel, institution-relevant claim
   (intra-week stress nowcasting) that the Fed's own weekly series cannot make.
4. **Success bar:** vintage OOS r ≥ 0.70 vs STLFSI4 **and** NBER AUC ≥ 0.80, with learned
   weights stable across decades.

### Track 2 — Calibrated, multi-market drawdown-hazard early-warning ⭐ the headline product

**Goal:** from "C-index 0.84, bad probabilities, one market" to "**calibrated** C-index
≈ 0.84 that **holds across ≥ 4 markets and ≥ 2 disjoint eras**."

1. Apply §2.1 + §2.2 calibration fixes → positive Brier skill.
2. Add §2.4 survival metrics (time-dependent AUC, IBS, competing-risks).
3. **Generalize across markets:** S&P 500, plus ≥3 of {FTSE 100, Nikkei 225, STOXX 600,
   TSX, Hang Seng}. Run the *identical* harness. If C-index ≈ 0.8 holds out-of-market,
   you have a generalization result that most published crisis models never demonstrate
   (this is literally your own Definition-of-Done criterion #5, currently unmet).
4. **Conformal prediction intervals** on every hazard output (distribution-free coverage).
   Institutions love guaranteed coverage; it's also your roadmap item #7.
5. **Success bar (your own DoD, upgraded):** calibrated (ECE < 0.03, Brier skill > 0),
   C-index ≥ 0.80 on **≥ 4 markets**, stable across **2 disjoint eras**.

### Track 3 — Reframe "multimodal" into a question that can actually win

The multimodal thesis fails *for forward-drawdown ranking*. Two reframes are defensible
and untested:

1. **Sentiment as a nowcasting input to the FSI, not a drawdown predictor.** Test whether
   real FinBERT daily sentiment improves the **STLFSI reconstruction** (Track 1), i.e. does
   news lower the FSI-vs-STLFSI tracking error between Fed releases? This is a *different*
   target (nowcasting a known index) where a weak signal can legitimately help, unlike a
   3.9% rare-event ranking problem.
2. **Sentiment as a hazard covariate, evaluated with time-dependent AUC.** Does adding
   sentiment raise AUC(t) specifically in the days *just before* onset? Pooled PR-AUC can
   hide a real short-window effect; the time-dependent metric is the honest place to look.

**Success bar:** sentiment improves FSI tracking error OR time-dependent AUC(t) in the
pre-onset window, under CPCV, with the change clearing a permutation/Granger test. If it
doesn't — report the second null cleanly. Either way you've asked a *new* question, not
re-litigated the old one.

---

## 5. How this beats the previous paper, concretely

| Dimension | Original Milestone (Group 13) | FCPS v4 (this roadmap) |
|---|---|---|
| Headline claim | F1 = 0.99 multimodal crisis prediction | Calibrated drawdown-hazard + daily Fed-STLFSI nowcast |
| Truth of headline | Leakage (circular target + smoothed regimes + event holdout) | Point-in-time, walk-forward, CPCV, calibrated |
| Evaluation | F1 on tiny all-positive windows | PR-AUC, Brier skill, ECE, IBS, time-dependent AUC, DSR, PBO |
| Baselines | none | base-rate, VIX, persistence — and an honest null vs VIX |
| Generalization | single market, in-sample | ≥ 4 markets, ≥ 2 eras, vintage data |
| Reproducibility | notebook | installable pkg, CI, pinned lockfile, committed artifacts |
| Honesty | overclaims | reports its own nulls and CIs |

The original paper's number was bigger. **Yours is real, calibrated, generalized, and
deployable.** In 2026 quant-ML reviewing — post the backtest-overfitting and TSFM-leakage
literature — *that* is the higher bar, and it's the bar institutions actually hire against.

---

## 6. Suggested paper / deck spine (the "institution-level" narrative)

1. **Hook:** "Most crisis-prediction results are leakage. We prove it on our own prior
   work, then build the honest version." (You have the receipts — use them.)
2. **Contribution 1 (positive):** daily, public-data, point-in-time nowcast of the Fed's
   STLFSI (r ≈ 0.7+ vintage), with learned-stable weights and intra-week lead time.
3. **Contribution 2 (positive):** a calibrated drawdown-hazard early-warning model,
   C-index ≈ 0.84, that generalizes across 4+ markets — with conformal coverage.
4. **Contribution 3 (deployable):** a stress-scaled overlay delivering statistically
   significant drawdown reduction (−18pp, CI clears zero), honestly *without* a Sharpe edge.
5. **Contribution 4 (methodological):** a rigorous null — under CPCV/DSR/PBO, neither
   credit, TDA, nor real-news sentiment beats a VIX threshold for drawdown ranking — a
   contribution to the overfitting literature.
6. **Limitations & ethics:** stated plainly (single asset class, EMH boundary, no causal
   claim on returns). This *section* is itself a credibility signal.

---

## 7. Prioritized backlog (do them in this order)

| Pri | Task | Effort | Payoff |
|---|---|---|---|
| **P0** | Drop `class_weight="balanced"` in `hazard.py` + isotonic-calibrate N-day incidence (§2.1) | hours | Flips your best result to deployable |
| **P0** | Vintage/ALFRED FRED + re-validate FSI vs STLFSI point-in-time (§2.3) | 1–2 days | Crown jewel becomes reviewer-proof |
| **P1** | Time-dependent AUC + IBS + competing-risks for hazard (§2.4) | 1–2 days | Survival eval at institution grade |
| **P1** | Multi-market hazard + FSI generalization, identical harness (Track 1.4 / 2.3) | 3–5 days | Meets your own DoD; rare in the literature |
| **P1** | Learned, stability-tested FSI weights (Track 1.2) | 1–2 days | Stronger than hand-set weights |
| **P2** | Regime-conditional + cost/capacity overlay reporting (§2.5) | 1–2 days | Overlay becomes a product claim |
| **P2** | Conformal intervals on hazard + API (Track 2.4) | 1–2 days | Coverage guarantees institutions expect |
| **P3** | Reframed multimodal experiments (Track 3) | 3–5 days | New question; either a win or a clean 2nd null |
| **P3** | Vectorize label loops; artifact provenance tags (§2.6) | hours | Scale + auditability |

**If you do only two things:** P0 (hazard calibration) and P0 (vintage FSI). Those two
turn your two best results from "promising" into "a desk would trust this," which is the
entire definition of institution-level.
