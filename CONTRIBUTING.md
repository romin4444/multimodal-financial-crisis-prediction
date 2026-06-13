# Contributing to FCPS

Thanks for considering a contribution. This project's distinguishing feature is
its commitment to **honest, leakage-free evaluation** of a multimodal crisis
forecaster — and that commitment only holds if changes are reviewed against the
same bar that produced the headline results.

## Quick start

```bash
git clone https://github.com/romin4444/multimodal-financial-crisis-prediction.git
cd multimodal-financial-crisis-prediction
pip install -e ".[dev]"
make test                 # 82 passing tests, ~15s
make demo                 # full v2 pipeline on cached synthetic data
python scripts/v3_run.py  # honest, leakage-free walk-forward on real S&P 500 + VIX
```

If any of those fail before you've made a change, please open an issue with the
error and your environment (`python --version`, OS, `pip freeze`).

## What kinds of changes are welcome

**Especially welcome:**

- New baselines for v3 (`src/v3/baselines.py`). The honest reading is "no model
  beats persistence" — counter-examples that hold up under purged walk-forward
  are exactly the contributions this project exists to find.
- New tests that pin a leakage-free behaviour (causality, look-ahead, scaler
  fit-window). See `tests/test_v3.py::TestCalibrationFix` for the style.
- Documentation fixes — especially anywhere the README disagrees with what the
  code actually does.
- New VIX-orthogonal features (`src/v3/vix_orthogonal.py`). Variance Risk
  Premium is currently the only one cleanly wired; term-structure and SKEW
  modules exist but need their input feeds.

**Will be pushed back on:**

- Anything that re-introduces in-sample evaluation, full-series scaler fits,
  or class-balanced classifiers without an explicit calibration story.
- Changes that touch `src/v3/walkforward.py` to *catch and ignore* per-fold
  failures (silent NaN OOS is the failure mode v3.1 explicitly fixed).
- Headline metric changes in the README without a reproducible artifact in
  `outputs/`.

## Development workflow

```bash
# 1. Branch from main
git switch -c your-change

# 2. Make changes and add/extend tests under tests/
#    Tests must run offline — no network, no FRED key required.

# 3. Run the gate that CI runs
python -m ruff check src tests scripts
python -m pytest -q

# 4. If you touched the v3 harness, also smoke-test the real pipeline
python scripts/v3_run.py    # writes outputs/v3_metrics.json
```

The CI gate lints `src/`, `tests/`, and `scripts/` only. Notebooks under
`notebooks/` and `legacy/` are intentionally excluded (they have unavoidable
Jupyter-style violations).

## Commit style

- Imperative subject ≤ 70 chars (`Fix FSI median leak under purged window`)
- Wrap the body at 72 cols; explain *why* over *what*
- One logical change per commit; never mix refactor + behaviour change
- If a commit changes a headline number, link to the `outputs/*.json` artifact

## Reporting bugs

Use the bug-report issue template. The single most useful thing you can give
me is a **minimum-reproducible script** under `scripts/repro_<issue>.py` — even
if it just calls `scripts/v3_run.py` with a different config. A claim that
"the numbers don't reproduce" without a script is very hard to action.

## Filing a feature request

Use the feature-request issue template. For new feature *families* (a new data
source, a new model class) please write a one-paragraph **why it would beat
the persistence baseline** before opening a PR — that's the gate that has
killed most of the candidates already tried (see `docs/EDGE_AND_MULTIMODAL_RESULTS.md`).

## License

By contributing you agree your changes are released under the MIT license that
covers the rest of the repository.
