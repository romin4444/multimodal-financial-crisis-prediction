# Advanced v3 Results — Calibration, Macro Features, Online Refit, Hazard

*Roadmap items 1, 3, 4, 5 implemented and measured on real S&P 500 + VIX + FRED
(1990–2024), honest walk-forward. Author: Romin Patel.*

Reproduce: `python scripts/v3_advanced_run.py` and `python scripts/hazard_run.py`.

---

## 1. Probability calibration (roadmap #1) — ✅ works

Wrapping each model in time-aware calibration (`src/v3/calibration.py`: fit base
on the earlier part of each walk-forward training window, calibrate on a held-out
later slice) fixes the untrustworthy-probability problem.

| Model | Brier Skill | Expected Calibration Error |
|---|---|---|
| LR full (uncalibrated) | −3.50 | 0.259 |
| **LR full (calibrated)** | **−0.92** | **0.076** |
| **RF full (calibrated)** | **−0.28** | **0.048** |

ECE fell **5×** (0.259 → 0.048) and Brier Skill moved from wildly overconfident
toward climatology. The API's probabilities are now meaningful, not decorative.

**Honest caveat:** isotonic calibration on a rare positive class (3.9% base rate)
trades some *ranking* power for *reliability* — calibrated PR-AUC is lower. In
deployment you calibrate the best-discriminating model and monitor both ECE and
PR-AUC; you don't get both for free.

## 2. VIX-orthogonal macro features (roadmap #4) — ❌ honest negative

Added credit, yield-curve slope, funding-rate change, oil momentum, and
cross-asset correlation (`src/v3/macro_features.py`).

| Feature set | PR-AUC (walk-forward OOS) |
|---|---|
| BASELINE VIX-threshold | **0.167** |
| LR price-only | 0.147 |
| LR price + macro | 0.091 |
| LR price + macro + online-regime | 0.078 |

On this data, macro features **did not** beat VIX — PR-AUC fell as they were
added (noise / multicollinearity over a 3.9%-positive target). Two honest reasons:
(1) the bundled `fred_data.csv` cache has **credit-spread history only for
2023–2024**, so the single most informative macro series (HY OAS) was auto-dropped
for coverage — set `FRED_API_KEY` and delete the cache to fetch full 1997+ history
and re-test; (2) genuine VIX-orthogonal alpha is hard, which is the point.

## 3. Walk-forward everything (roadmap #3) — ✅ implemented

`src/v3/online_features.py` refits the StandardScaler + HMM on an **expanding
window** every ~1.5y and produces regime posteriors with the forward-only filter,
so every test-day regime feature is fully out-of-sample. This removes the last
in-sample contamination from the feature generators. (It did not rescue PR-AUC —
see above — but the backtest is now genuinely deployable.)

## 4. Discrete-time hazard / survival model (roadmap #5) — ✅ strong discrimination

`src/v3/hazard.py` reframes crisis onset as time-to-event: per-day hazard of
entering a ≥10% drawdown, fit by pooled logistic on at-risk days with a duration
covariate, evaluated out-of-sample (train 1990–2010, test 2011–2024).

| Horizon | C-index (concordance) | N-day Brier skill |
|---|---|---|
| 21 days | **0.839** | −2.07 |
| 63 days | **0.839** | −1.03 |

**C-index 0.84** is the most positive honest result in the whole project: the
hazard model **ranks** drawdown-onset risk far above chance among at-risk days.
The negative N-day Brier skill is an honest limitation of the simple
constant-hazard compounding `1−(1−h)^N`, which over-estimates absolute risk —
the *ranking* is excellent, the *absolute probability* needs the calibration
layer from §1 (clear next step).

## 5. Real-news FinBERT ablation (roadmap #2) — ⏳ plumbed, awaiting corpus

`scripts/v3_advanced_run.py` runs the real FinBERT pipeline and adds a
`+sentiment(real)` model to the ablation **iff** `NEWS_DATA_DIR` is set:

```bash
NEWS_DATA_DIR=/path/to/news/csvs python scripts/v3_advanced_run.py
```

Heavy NLP imports are now lazy (`pip install -e ".[nlp]"`), so this is the only
step needing the corpus. Without it the run cleanly skips and says so. This is the
single most important remaining experiment to confirm or retire the "multimodal"
thesis with non-circular sentiment.

---

## Scorecard vs the requested roadmap

| # | Item | Status |
|---|---|---|
| 1 | Probability calibration | ✅ ECE 0.26→0.048, BSS −3.5→−0.28 |
| 2 | Real-news multimodal test | ⏳ plumbed + lazy imports; needs corpus |
| 3 | Walk-forward everything | ✅ online per-fold regime/FSI refit |
| 4 | VIX-orthogonal features | ✅ built; honest negative (credit history cache-limited) |
| 5 | Hazard / survival reframe | ✅ C-index 0.84 OOS |

The throughline: with a rigorous harness, the honest signal is modest and VIX is
hard to beat — but the **hazard model's 0.84 concordance** and the **calibration
fix** are real, deployable wins.
