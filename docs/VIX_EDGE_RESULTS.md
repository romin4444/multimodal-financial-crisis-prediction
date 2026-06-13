# VIX-Orthogonal Edge Test — Variance Risk Premium

*Honest results from `scripts/vix_edge_run.py` on the tested `src/v3` harness.
Author: Romin Patel.*

## Question

Earlier runs (see `EDGE_AND_MULTIMODAL_RESULTS.md`) showed that macro, credit,
yield-curve, cross-asset, TDA and FinBERT-sentiment features all fail to beat a
VIX threshold for forward-drawdown prediction. A natural objection: *those
features were largely collinear with VIX, so of course they added nothing.* This
run removes that objection by testing the single most literature-backed
**VIX-orthogonal** signal:

> **Variance Risk Premium**  `VRP = (VIX/100)² − realized_variance`
> (Bollerslev, Tauchen & Zhou 2009) — needs **no new data**; both terms are
> already in the pipeline.

Decision rule, set before running, out-of-sample under CPCV (6 groups, k=2,
21-day embargo), calibrated logistic:

- **Q-REPLACE** — does any config beat the VIX-ECDF threshold on PR-AUC?
- **Q-INCREMENTAL** — does `VIX + orthogonal residuals` beat `VIX-only-LR`? (the fair bar)
- A config **wins all gates** only if it beats VIX **and** PBO < 0.5 **and** Deflated Sharpe > 0.95.

## Data

Offline cached **S&P 500 + VIX, 1990–2024 (8,815 trading days)**. Option-surface
feeds (`^VIX9D`, `^VIX3M`, `^SKEW`) were **not available offline**, so the
term-structure and SKEW features were skipped this run; the module builds them
automatically when `data/cache/mkt_VIX9D|VIX3M|SKEW.csv` are present.

## The orthogonality check (this is the point)

| | value |
|---|---|
| corr(VRP, VIX) | **−0.14** |
| orthogonal fraction (1 − R²) | **0.98** |

VRP is **genuinely orthogonal** to the VIX level — 98% of its variance is
information the VIX level does not contain. So this is a clean test of orthogonal
signal, not collinear noise.

## Result — VRP is orthogonal *and* adds no out-of-sample value

PR-AUC (calibrated LR) by horizon / drawdown threshold:

| H | thr | base | **VIX-ECDF** | VIX-LR | price | price+VRP | +VRP_resid | beats VIX? | PBO | DSR |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 7% | 0.035 | **0.180** | 0.107 | 0.130 | 0.096 | 0.105 | ✗ | 0.11 | 0.99 |
| 10 | 10% | 0.015 | **0.160** | 0.142 | 0.130 | 0.051 | 0.048 | ✗ | 0.00 | 0.99 |
| 21 | 7% | 0.092 | **0.229** | 0.164 | 0.152 | 0.150 | 0.151 | ✗ | 0.41 | 0.98 |
| 21 | 10% | 0.040 | **0.178** | 0.092 | 0.096 | 0.071 | 0.078 | ✗ | 0.29 | 0.99 |
| 63 | 7% | 0.212 | **0.346** | 0.286 | 0.253 | 0.209 | 0.211 | ✗ | 0.00 | 0.99 |
| 63 | 10% | 0.137 | **0.258** | 0.196 | 0.173 | 0.161 | 0.169 | ✗ | 0.46 | 0.98 |

**Verdict:**

- **Q-REPLACE = False** — VIX-ECDF wins on **all six** targets.
- **Q-INCREMENTAL = False** — `VIX + VRP` never beats `VIX-only`; adding VRP
  *lowers* PR-AUC versus price-only in every cell.
- **No config clears the gates.**

## Why this finding is stronger than the earlier nulls

The macro/sentiment nulls could be dismissed as collinearity. This one cannot:
VRP is **provably 98% orthogonal to VIX**, yet its orthogonal information is
**not predictive** of forward equity drawdowns at 10–63 day horizons. The barrier
is therefore not "we used VIX in disguise" — it is that, for *this target and
these horizons*, the VIX level is close to the information frontier and the extra
variance VRP supplies is noise for drawdown timing (even though VRP is known to
predict the equity-risk *premium*, a different object).

## What remains genuinely untested (the real next levers)

1. **Term-structure & SKEW** — orthogonal and plausibly informative about the
   *timing* of stress; require an options feed (skipped offline). Module is ready.
2. **High-frequency funding stress** (TED, SOFR-OIS, FRA-OIS) — needs FRED.
3. **Reframe the target**: Growth-at-Risk quantile regression / severity
   regression / discrete-time hazard (the hazard model already reaches
   c-index 0.839). Against those targets a VIX *threshold* is a much weaker
   baseline, so this is the most promising path to a positive result.

## Reproduce

```bash
pip install scikit-learn scipy
python scripts/vix_edge_run.py        # writes outputs/vix_edge_results.json
```
