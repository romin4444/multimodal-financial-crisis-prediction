# FCPS — Financial Crisis & Stock-Direction Prediction System

[![CI](https://github.com/romin4444/multimodal-financial-crisis-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/romin4444/multimodal-financial-crisis-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Author: Romin Patel**  ·  *MBAI 5600G capstone (Group 13), rebuilt into a modular, tested, leakage-free pipeline.*

A multimodal pipeline for financial-market **crisis detection** and **stock-price
direction detection**, fusing price regimes (HMM), conditional volatility (GARCH),
macro stress (FRED), and news sentiment (FinBERT), with SHAP explainability — and,
above all, an **honest, leakage-free, walk-forward evaluation harness**.

---

## ⭐ The honest headline (read this first)

Most "crisis prediction" code reports stellar in-sample numbers that evaporate
out-of-sample. This project's distinguishing feature is that it **measures and
reports that gap instead of hiding it.**

| | In-sample (naive) | **Honest walk-forward (v3)** |
|---|---|---|
| Crisis F1 | 0.99 | — |
| Crisis PR-AUC (out-of-sample) | — | **~0.10** (a VIX threshold scores 0.17) |
| 5-day stock direction, edge over "always-up" | — | **−0.05** (no reliable edge — consistent with weak-form EMH) |

The one result that **survives** rigorous scrutiny: our daily **Financial Stress
Index reproduces the St. Louis Fed's published STLFSI at r = 0.823** (NBER
recession AUC = 0.861) using only public data — a genuinely deployable signal.

Full diagnosis & roadmap: [`docs/PROJECT_REVIEW_AND_ROADMAP.md`](docs/PROJECT_REVIEW_AND_ROADMAP.md).
Direction study: [`docs/DIRECTION_RESULTS.md`](docs/DIRECTION_RESULTS.md).

---

## 🚀 Try it in 30 seconds (no API key needed)

```bash
git clone https://github.com/romin4444/multimodal-financial-crisis-prediction.git
cd multimodal-financial-crisis-prediction
pip install -e .
make demo            # or: python scripts/demo_run.py
```

`make demo` runs the full pipeline end-to-end on cached/synthetic data — no FRED
key, no internet, no GPU — and writes figures + metrics to `outputs/`.

---

## Two evaluation tracks

| Track | Script | What it does |
|---|---|---|
| **v2** (original pipeline) | `scripts/real_data_run.py` | Crisis pipeline on real S&P 500 + VIX + FRED. Strong *in-sample* numbers. |
| **v3** (honest) | `scripts/v3_run.py` | Leakage-free: exogenous forward-drawdown target, **causal** (forward-only) regime probabilities, **purged walk-forward**, real baselines, calibration. |
| **v3 direction** | `scripts/direction_run.py` | Next-N-day up/down detection per ticker (AAPL, JPM, XOM, GS, ^GSPC) on the same honest harness. |
| **v3 advanced** | `scripts/v3_advanced_run.py` | Adds probability calibration, VIX-orthogonal macro features, and online (per-fold refit) regime/FSI. |
| **hazard** | `scripts/hazard_run.py` | Discrete-time survival model: P(≥10% drawdown within N days). |

---

## Installation

```bash
pip install -e .              # core (no heavy NLP) — enough for demo, v3, API, tests
pip install -e ".[nlp]"       # + FinBERT (torch, transformers) for real news sentiment
pip install -e ".[explain]"   # + SHAP explainability
pip install -e ".[all]"       # everything (incl. dev/test tooling)

# Exact reproducible versions used for the reported numbers:
pip install -r requirements.lock
```

Optional config for the full pipeline:
```bash
cp .env.example .env          # add FRED_API_KEY (free at fred.stlouisfed.org)
# Optional: NEWS_DATA_DIR=/path/to/news/csvs  for real FinBERT sentiment
```

---

## Repository structure (actual)

```
multimodal-financial-crisis-prediction/
├── README.md                  • this file
├── LICENSE                    • MIT (© Romin Patel)
├── pyproject.toml             • installable package (pip install -e .)
├── requirements.txt           • flexible deps   |  requirements.lock = exact pins
├── Makefile                   • make demo / test / serve / v3 / direction
├── config.yaml                • all parameters (Pydantic-validated)
├── .github/workflows/ci.yml   • GitHub Actions: install + pytest + ruff
├── fred_data.csv              • cached FRED snapshot (lets the pipeline run key-free)
│
├── src/
│   ├── config.py              • Pydantic config loader (singleton: cfg)
│   ├── logging_setup.py       • structured JSON logging
│   ├── pipeline.py            • v2 orchestration
│   ├── data/                  • market.py · fred.py · news.py
│   ├── features/              • engineering.py (price features + diagnostics)
│   ├── models/                • fsi.py · garch.py · hmm.py · sentiment.py · fusion.py
│   ├── analysis/              • lead_lag.py · shap_explain.py · benchmarks.py
│   ├── evaluation/            • validation.py
│   ├── visualization/         • plots.py (8 figures)
│   ├── api/                   • app.py (FastAPI) · schemas.py
│   └── v3/                    ★ leakage-free harness
│       ├── labeling.py        • exogenous forward-drawdown target
│       ├── causal_regime.py   • forward-only (filtered) HMM posteriors
│       ├── walkforward.py     • purged, embargoed expanding-window backtest
│       ├── baselines.py       • base-rate / VIX / persistence baselines
│       ├── metrics.py         • PR-AUC, Brier skill, ECE, lift, economic backtest
│       ├── calibration.py     • time-series probability calibration
│       ├── macro_features.py  • VIX-orthogonal features (credit, curve, funding)
│       ├── online_features.py • per-fold refit regime/FSI (walk-forward everything)
│       ├── direction.py       • stock up/down detection
│       └── hazard.py          • discrete-time survival model
│
├── scripts/                   • demo_run · real_data_run · v3_run · v3_advanced_run
│                                · direction_run · hazard_run · train · serve
├── tests/                     • pytest suite (incl. look-ahead/causality proofs)
├── docs/                      • PROJECT_REVIEW_AND_ROADMAP.md · DIRECTION_RESULTS.md
├── outputs/                   • generated figures + metrics JSON + reports
└── legacy/                    • original notebooks, final.py monolith, milestone PDFs
```

---

## API

```bash
make serve            # uvicorn on http://localhost:8000  (docs at /docs)
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health + last data date |
| `GET` | `/metrics` | Model evaluation metrics |
| `POST` | `/predict` | Single-date crisis probability |
| `POST` | `/predict/batch` | Date-range batch predictions |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"target_date": "2020-03-01", "horizon_days": 5}'
```

---

## Methods

**Quantitative:** ARMA-GARCH / GJR / EGARCH (BIC selection) · Financial Stress Index
(VIX + GARCH + drawdown + credit) · 3-state Gaussian HMM regimes.
**NLP:** FinBERT (GPU FP16) · VADER baseline · synthetic VIX proxy (clearly flagged).
**Fusion & XAI:** Logistic / RF / GBM ensemble · SHAP TreeExplainer · lead-lag
cross-correlation · Granger causality.
**v3 rigor:** exogenous labels · causal filtered regimes · purged walk-forward ·
probability calibration · honest baselines · economic backtest.

---

## Testing & CI

```bash
make test             # full pytest suite (56 tests)
```

GitHub Actions runs `pip install -e ".[dev]"` + `pytest` + `ruff` on every push
(Python 3.10 and 3.12). Heavy NLP deps are optional and their tests skip cleanly
when torch isn't installed, so CI stays fast and green.

---

## Reproducibility

- `requirements.lock` pins the exact versions behind every reported number.
- `fred_data.csv` is a committed FRED snapshot so the pipeline runs without an API key.
- All randomness is seeded (`config.yaml: seed`).
- `make demo` reproduces the headline artifacts in ~30 s.

---

## License

MIT — see [LICENSE](LICENSE). © 2026 **Romin Patel**.

This is academic capstone work; if you reuse it, attribution to Romin Patel is appreciated.
