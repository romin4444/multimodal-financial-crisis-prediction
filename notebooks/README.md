# notebooks/ — Kaggle frontier benchmark

**`kaggle_frontier_benchmark.ipynb`** — a self-contained, GPU-ready benchmark that
evaluates crisis prediction at the 2025–2026 methodological bar. Author: Romin Patel.

## Run on Kaggle (recommended — GPU)

1. Create a new Kaggle Notebook → upload `kaggle_frontier_benchmark.ipynb`.
2. **Settings → Accelerator → GPU T4 x2**, **Internet → ON**.
3. *(Optional)* Add a FRED key: **Add-ons → Secrets → `FRED_API_KEY`** (free at
   fred.stlouisfed.org) to enable the credit/yield macro features.
4. *(Optional)* Attach any news dataset (date + headline columns) as input to
   enable the **real FinBERT** sentiment ablation.
5. Run all. Outputs land in `/kaggle/working/`:
   - `frontier_benchmark_metrics.json` — the full honest results
   - `frontier_benchmark.png` — PR-AUC comparison vs the VIX baseline

## What it runs (one honest harness)

Exogenous forward-drawdown target · causal filtered HMM regime · **CPCV** +
embargo · **PBO** + **Deflated Sharpe** · calibration (ECE/Brier) · honest
baselines (VIX / persistence / base-rate) · **TDA** persistent-homology features ·
**Chronos** time-series foundation model (zero-shot) · optional **FinBERT** ·
discrete-time **hazard** model.

Optional pieces (TDA / Chronos / FinBERT / FRED) self-skip cleanly if their
library/data/secret is absent — the core benchmark always completes.

## Files
- `kaggle_frontier_benchmark.py` — the notebook as a percent-script (source of truth).
- `kaggle_frontier_benchmark.ipynb` — generated notebook (upload this to Kaggle).
- `_py_to_ipynb.py` — dependency-free `.py → .ipynb` converter.
- `_validate_local.py` — runs the notebook's core logic locally (CPU) as a smoke test.

Regenerate the `.ipynb` after editing the `.py`:
```bash
python notebooks/_py_to_ipynb.py
```
