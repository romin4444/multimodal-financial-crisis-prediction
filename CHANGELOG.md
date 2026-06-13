# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project tries (loosely) to follow [Semantic Versioning](https://semver.org/).

## [3.3.0] — 2026-06-13

The "v4 institutional roadmap, P0 items shipped" release. An independent
review (`docs/INSTITUTIONAL_ROADMAP_V4.md`) identified the hazard model as
the project's strongest result hobbled by the exact `class_weight="balanced"`
bug v3.1 fixed elsewhere — and called for vintage/ALFRED FRED data to make
the crown-jewel FSI result reviewer-proof. v3.3 ships the hazard fix and
scaffolds the vintage pathway.

### Added
- `src/v3/hazard.py::HazardFit.incidence_calibrator` — isotonic calibration of
  the N-day cumulative incidence, fit on a held-out 20% slice of the training
  mask (per the v4 roadmap §2.1 prescription: calibrate the *N-day risk*, not
  the per-step hazard). `fit_hazard(..., horizon=N, calibrate=True)` attaches
  it automatically; `cumulative_incidence(X, horizon, calibrated=True)`
  returns the calibrated probability when available.
- `evaluate_hazard` now reports **both** the raw and the isotonic-calibrated
  Brier / Brier-skill so the calibration uplift is visible.
- `src/data/fred.py::download_fred_vintage(as_of=...)` and the
  `realtime_start`/`realtime_end` plumbing in `_fetch_series` — ALFRED
  point-in-time fetch with a per-as-of cache under
  `data/cache/fred_vintage/<YYYY-MM-DD>.csv`.
- `scripts/fsi_vintage_validate.py` — runs the FSI vs STLFSI comparison on
  a grid of as-of dates and writes `outputs/fsi_vintage_validation.json`.
  Without `FRED_API_KEY` it exits 0 with a clear message so CI stays green.
  The v4 success bar (`--success-r 0.70`) is checked per as-of row.
- `notebooks/kaggle_fcps_v4_results.ipynb` — single Kaggle cell that
  reproduces the three results that actually survive (FSI nowcast, calibrated
  hazard, risk overlay) with the v4 institutional framing.
- `docs/INSTITUTIONAL_ROADMAP_V4.md` — the independent review and prioritized
  backlog that drove this release.
- `tests/test_v3_advanced.py::TestHazard::test_no_class_weight_balanced_in_hazard_fit`
  — regression guard so the §2.1 fix can't be silently undone.
- `tests/test_v3_advanced.py::TestHazard::test_calibrated_incidence_is_within_unit_interval`
  — pins the isotonic-calibrator API + the no-inversion invariant.

### Changed
- **`src/v3/hazard.py`** drops `class_weight="balanced"` from the
  `LogisticRegression`. On real S&P 500 1990–2024, this alone moves the
  21-day Brier skill from the originally-reported **−2.07** to **−0.22**
  (raw), with the calibrated 63-day version reaching **−0.11** — a ~10×
  improvement, exactly the v3.1 fusion-calibration story repeated.
- `_realized_within` vectorized via cumulative sum — O(n) instead of the
  prior O(n·horizon) Python loop (per v4 roadmap §2.6).
- `scripts/hazard_run.py` now refits per horizon (the per-step hazard model
  is horizon-independent but the isotonic calibrator IS horizon-specific),
  reports BSS_raw and BSS_calibrated side by side, and stores both in
  `outputs/hazard_metrics.json`.
- `src/models/fusion.py`: added a multi-line code comment above the v2
  `class_weight="balanced"` block pointing at the v3.1 fusion fix, the v3.3
  hazard fix, and roadmap §2.6, so nobody copies the legacy pattern forward.

## [3.2.0] — 2026-06-13

The "deployable result" release. v3.1 made the crisis-probability output
trustworthy; v3.2 ships the actual end-use case the project has always been
heading toward — a leakage-free risk overlay that converts the rankable-risk
finding into a real reduction in drawdown.

### Added
- `scripts/risk_overlay_run.py` — stress-scaled risk overlay on SPY:
  vol-targeting + risk-off cut when an *expanding-window* stress z-score
  (VIX + trailing RV + drawdown) is in its top decile. Position at t+1
  uses only data through close of t (`.shift(1)` + asserted alignment).
  Reports: bootstrap CI on Sharpe-diff and max-drawdown reduction
  (Politis–Romano stationary block bootstrap, B = 2000, expected block
  21d), and Deflated Sharpe Ratio (Bailey & López de Prado 2014) against
  the vol-target grid we searched.
  On real 1993–2026 SPY+VIX: drawdown −55.2% → −37.1%
  (95% CI [+5.9pp, +36.2pp], significant); Sharpe edge +0.088
  (95% CI [−0.086, +0.273], not significant, EMH-consistent).
  Kaggle-portable (single self-contained file, synthetic fallback when
  offline). `make risk-overlay` target added.
- `tests/test_risk_overlay.py` — 9 regression tests pinning the three
  properties the overlay depends on: weights are causal (TestCausalWeights
  perturbs the future and asserts today's weight is byte-identical),
  bootstrap is reproducible under a fixed seed, and per-period Sharpe
  stays below 0.20 (anything higher = suspected lookahead leak).

### Changed
- README adds the risk-overlay headline table and a row in the evaluation-
  tracks table. The deployable result is now stated explicitly above the fold.

## [3.1.0] — 2026-06-13

The "calibration honesty" release. Out-of-sample ranking metrics in v3 were
trustworthy, but the *probabilities* a downstream user actually consumes were
not. v3.1 fixes that and tightens the harness so the loud failure modes are
loud.

### Added
- `src/json_utils.py` — shared `safe_json_default` encoder so every script's
  output JSON serializes `numpy.bool_` as a real boolean (no more
  `"target_met": "True"`), `NaN`/`Inf` as `null`, and rejects `pd.Series`.
- `tests/test_v3.py::TestCalibrationFix` — regression tests pinning Brier
  skill > 0 and ECE < 0.05 for an un-weighted LR on a rare-event base rate,
  and ECE > 0.10 for the balanced variant. The fix can't silently regress.
- `src/v3/walkforward.WalkForwardError` — the harness now raises if zero folds
  succeed instead of returning an all-NaN OOS series.
- `.gitattributes` — mark `*.ipynb` as vendored, `legacy/**` as documentation,
  `outputs/**` as generated, so GitHub's language bar reflects the actual
  Python package.
- `CONTRIBUTING.md`, `CITATION.cff`, issue + PR templates, this changelog.
- README hero image (FSI vs STLFSI) above the fold; topic badges.

### Changed
- **Calibration fix.** Dropped `class_weight="balanced"` from `LogisticRegression`
  and `RandomForestClassifier` in `scripts/v3_run.py`. On the ~4% base rate this
  was inflating every predicted probability ~20×, producing Brier skill ≈ -3
  and ECE ≈ 0.31. The un-weighted variant moves Brier skill positive (climatology-
  beating) and ECE down to ≈ 0.017, with PR-AUC unchanged-or-better. The original
  balanced variant is retained as a parallel contender (`MODEL price-only (LR,
  balanced)`) so the before/after is visible on every run.
- **Economic-backtest threshold leak.** Was `best_oos.quantile(0.85)` over the
  *entire* OOS series — a forward quantile of our own future predictions. Now
  the 0.85 quantile is calibrated on the first half of OOS and applied only
  to the second half.
- **FSI scaler leaks.** `src/models/fsi.py::_scale` now imputes NaNs with the
  *train-only* median (was full-series); `update_with_garch` reuses the
  `train_mask` from `build()` instead of refitting the GARCH MinMaxScaler on
  the full series.
- `scripts/v3_run.py` now uses `src/v3/online_features.compute_online_regime`
  (per-fold refit HMM + scaler) instead of the fit-once-on-train-head shortcut.
  That module previously existed as dead code claiming to be "the deployable
  version".
- `scripts/demo_run.py` ends with a loud caveat that its F1 ≈ 1.0 numbers are
  the leaky v2 in-sample result and points users at `scripts/v3_run.py`.
- README headline rewritten as a v3 (balanced/mis-calibrated) vs v3.1
  (calibrated) before/after table. The `r = 0.823` headline softened to the
  reproducible `r ≈ 0.76–0.82` range. Test count corrected from 71 to 82.
  CI lint coverage clarified (`src/ tests/ scripts/` only; notebooks excluded).
- README `src/v3/` listing now reflects the actual 15 modules (previously
  named 10).

### Removed
- 5 near-duplicate `legacy/notebooks/notebook9a5760bcc1*.ipynb` files (~2.1 MB)
  that were the largest contributor to GitHub mis-labelling the repo as
  "83% Jupyter Notebook".
- 8 pre-rendered figure PNGs (≈2.5 MB), `outputs/FCPS_Report.pdf` (3.5 MB),
  and `FCPS_vs_Institutions.pptx` (0.5 MB) from version control. They are
  regenerable from source and now ship as GitHub release assets. The hero
  figure was kept at `docs/hero_fsi_vs_stlfsi.png` for the README only.

## [3.0.0] — 2026-06

The "honest harness" release. Adds `src/v3/`: exogenous forward-drawdown
labels, causal (forward-only) filtered HMM posteriors, purged + embargoed
walk-forward, real baselines (base-rate / VIX / persistence), CPCV +
Deflated Sharpe + PBO gating, and the FastAPI service.

## [2.x] — 2026-05

The original capstone pipeline: HMM + GARCH + FinBERT + FSI fused via
logistic / RF / GBM, with SHAP explainability, evaluated on event-based
holdouts over GFC / COVID / Inflation crisis windows. In-sample F1 ≈ 0.99
on those windows — a number v3 explicitly exists to put in honest context.
