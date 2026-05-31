# Stock Price Direction Detection — Results

*Task pivot: from crisis prediction → next-5-day stock direction (up/down).*
*Evaluated with the v3 leakage-free walk-forward harness. Real data, 1990–2024.*

---

## What was built

| Component | File |
|---|---|
| Direction target + per-stock features + directional metrics + economic backtest | `src/v3/direction.py` |
| Directional baselines (always-up, momentum) | `src/v3/direction.py` |
| Multi-stock walk-forward runner | `scripts/direction_run.py` |
| 11 unit tests | `tests/test_direction.py` |
| Results JSON | `outputs/direction_metrics.json` |

**Target:** `1 if close[t+5] > close[t] else 0` per stock (exogenous, causal).
**Features:** 11 per-stock technicals (momentum, vol, RSI, MA-distance, drawdown, volume) + 5 market-context features (VIX, FSI, causal index-regime probs, market momentum).
**Models:** Logistic (price-only), Logistic (price+market), Random Forest (full).
**Baselines:** always-up, momentum, base-rate.
**Eval:** expanding-window walk-forward, 126-day refit, 5-day embargo. Stocks: AAPL, JPM, XOM, GS, ^GSPC.

---

## Headline result

```
Mean OOS accuracy EDGE of ML models over 'always-up' : -0.0494
Interpretation: No reliable edge — consistent with weak-form EMH.
```

No ML model beat the trivial "always predict up" baseline out-of-sample on any of the five names. AUCs cluster around 0.47–0.54 (chance = 0.50); Matthews correlation ≈ 0.

| Stock | Always-up acc | Best ML acc | ML edge | Best ML AUC |
|---|---|---|---|---|
| AAPL | 0.558 | 0.532 | −0.026 | 0.510 |
| JPM | 0.550 | 0.501 | −0.049 | 0.491 |
| XOM | 0.547 | 0.523 | −0.025 | 0.537 |
| GS | 0.543 | 0.497 | −0.046 | 0.497 |
| ^GSPC | 0.581 | 0.502 | −0.079 | 0.505 |

---

## Why this is the *correct* result, not a failure

Short-horizon direction of liquid stocks being unpredictable is the single most
replicated finding in finance — weak-form market efficiency (Fama 1970), the
random-walk hypothesis (Malkiel 1973), and decades of failed technical-analysis
backtests. A pipeline that reported 90%+ directional accuracy would be a **red
flag for leakage**, exactly the trap v2's crisis model fell into.

The value here is that the v3 harness **refuses to manufacture fake skill**:
- Exogenous label (forward return sign) — no circular target.
- Causal features only — no future information.
- Walk-forward + embargo — no peeking.
- Baselines + honest metrics — the bar is "beat always-up," and we don't.

This is the same lesson as the crisis study, now confirmed on a second task.

---

## One nuance: the economic backtest

Trading the signal (long when P(up) ≥ 0.5, else cash) vs buy-and-hold:

| Stock | Strategy Sharpe | Buy&Hold Sharpe | Verdict |
|---|---|---|---|
| AAPL | 0.55 | 0.75 | B&H wins |
| JPM | 0.44 | 0.53 | B&H wins |
| **XOM** | **0.70** | **0.51** | **Strategy wins** |
| GS | 0.40 | 0.45 | B&H wins |
| ^GSPC | 0.36 | 0.55 | B&H wins |

XOM's de-risking strategy beat buy-and-hold on a risk-adjusted basis — but this
is **1 of 5** and almost certainly sampling noise, not a durable edge. Acting on
it would be exactly the kind of overfitting the harness is designed to expose.
For the upward-drifting names, sitting in cash simply misses the drift.

---

## How to reproduce

```bash
python scripts/direction_run.py        # ~2–3 min on cached data
python -m pytest tests/test_direction.py
```

## If the goal is to actually find directional edge (future work)

The honest negative result points the way:
1. **Longer horizons / lower-frequency** signals (monthly, quarterly) where drift
   and value/momentum factors are documented to work.
2. **Cross-sectional ranking** (long top-decile, short bottom-decile) rather than
   single-name timing — this is where quant equity actually makes money.
3. **Alternative data** orthogonal to price (earnings revisions, real FinBERT news
   sentiment, supply-chain, options flow).
4. **Risk-targeting** rather than direction-timing — vol forecasting (our GARCH)
   has real, documented skill; direction does not.
