# notebooks/ — Kaggle GPU notebooks

Author: Romin Patel. Two self-contained, GPU-ready notebooks:

### `kaggle_edge_and_multimodal.ipynb`  ← run this to close the two open questions
Focused test of (Q1) **can any model beat a VIX threshold out-of-sample?** and
(Q2) **does real FinBERT news sentiment add value?** Sweeps the target over
horizons × thresholds, adds VIX-orthogonal macro + TDA + real-news sentiment, and
judges each config by `PR-AUC > VIX` **and** `PBO < 0.5` **and** `Deflated Sharpe > 0.95`.
**Attach a `FRED_API_KEY` secret and a financial-news dataset for the full test.**
Output: `edge_multimodal_results.json` (send this back).

### `kaggle_frontier_benchmark.ipynb`
The broad benchmark that evaluates crisis prediction at the 2025–2026 bar
(CPCV + PBO + DSR + TDA + Chronos TSFM + FinBERT + hazard, all in one harness).

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
