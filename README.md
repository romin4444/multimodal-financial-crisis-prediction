# Financial Crisis Prediction System (FCPS)
**MBAI 5600G | Group 13 | Jeya Surya Balaji & Keertan Patel**

Multimodal early-warning system for financial market crises.  
Fuses price regimes (HMM), conditional volatility (GARCH), and news sentiment (FinBERT) into an interpretable ensemble classifier with SHAP explainability.

> **Two evaluation tracks:**
> - **v2 (`scripts/real_data_run.py`)** — the original crisis pipeline. Strong *in-sample* numbers, but see the honest re-evaluation below.
> - **v3 (`scripts/v3_run.py`, `scripts/direction_run.py`)** — leakage-free, walk-forward evaluation with exogenous targets and real baselines. This is the trustworthy track. See `docs/PROJECT_REVIEW_AND_ROADMAP.md`.
>
> **Tasks supported:** (a) crisis detection (forward-drawdown target) and (b) **stock price direction detection** (up/down over N days, per ticker) — see `docs/DIRECTION_RESULTS.md`.

---

## Architecture

```
Data Layer          Feature Layer      Model Layer           Output Layer
──────────────      ─────────────      ───────────           ────────────
yfinance (OHLCV) ─► Feature Eng.  ─► FSI (composite)  ─► Fusion Model
FRED REST API    ─► GARCH Var.    ─► GARCH(1,1)/GJR   ─► SHAP Explain
News CSVs        ─► FinBERT Sent  ─► HMM (3 states)   ─► REST API
                                   ► Lead-Lag Analysis ─► 8 PNG charts
                                                        ─► metrics.json
```

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env — add your FRED API key (free at fred.stlouisfed.org)
```

### 3. Train
```bash
python scripts/train.py
# With custom news directory:
python scripts/train.py --news-dir /path/to/news/csvs
```

### 4. Serve
```bash
python scripts/serve.py
# API docs: http://localhost:8000/docs
```

### 5. Test
```bash
make test
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health + last data date |
| `GET` | `/metrics` | Model evaluation metrics |
| `POST` | `/predict` | Single-date crisis probability |
| `POST` | `/predict/batch` | Date-range batch predictions |

### Predict example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"target_date": "2020-03-01", "horizon_days": 5}'
```

---

## Key enterprise upgrades (vs. original `final.py`)

| Area | Original | Upgraded |
|------|----------|----------|
| Architecture | 2,016-line monolith | Modular `src/` package |
| Configuration | Hardcoded Python constants | `config.yaml` + Pydantic validation |
| Data leakage | `MinMaxScaler` fit on full data | Fit on training mask only |
| Threshold tuning | Training set → overfit | Dedicated validation split |
| Paths | Kaggle-specific `/kaggle/working/` | Portable, env-var configurable |
| FRED key | Kaggle Secrets API only | `FRED_API_KEY` env var (any platform) |
| Model persistence | Raw pickle | joblib + metadata JSON |
| Logging | `print()` + basic logging | Structured JSON logs (rotation) |
| API | None | FastAPI REST with Pydantic schemas |
| Tests | None | pytest suite with synthetic fixtures |
| Forward-fill | Unlimited propagation | Capped at 22 trading days |
| Synthetic sentiment | Potential look-ahead | Verified backward-looking only |
| FRED key loading | Kaggle-only | Platform-agnostic env var |

---

## Configuration

Edit `config.yaml` — all parameters are documented inline. Key sections:

- **`data`** — tickers, date range, FRED series
- **`fsi`** — component weights (must sum to 1.0)
- **`hmm`** — state count candidates, EM seeds
- **`finbert`** — model name, batch size, panic threshold
- **`fusion`** — horizon, feature columns, hyperparameters
- **`crisis_windows`** — GFC 2008, COVID 2020, Inflation 2022
- **`api`** — host, port, worker count

---

## Project structure

```
keertan/
├── config.yaml              # All parameters
├── requirements.txt
├── pytest.ini
├── Makefile
├── .env.example
├── src/
│   ├── config.py            # Pydantic config loader (singleton: cfg)
│   ├── logging_setup.py     # Structured JSON logging
│   ├── pipeline.py          # Main orchestration
│   ├── data/
│   │   ├── market.py        # yfinance download
│   │   ├── fred.py          # FRED REST API
│   │   └── news.py          # News CSV auto-detection
│   ├── features/
│   │   └── engineering.py   # Price features + diagnostics
│   ├── models/
│   │   ├── fsi.py           # Financial Stress Index (leakage-free)
│   │   ├── garch.py         # ARMA-GARCH BIC selection
│   │   ├── hmm.py           # Gaussian HMM regime detection
│   │   ├── sentiment.py     # FinBERT + VADER + synthetic proxy
│   │   └── fusion.py        # Multimodal ensemble + threshold tuning
│   ├── analysis/
│   │   ├── lead_lag.py      # Cross-correlation + Granger causality
│   │   ├── shap_explain.py  # SHAP TreeExplainer
│   │   └── benchmarks.py    # Wang2025 + FinBERT vs VADER
│   ├── evaluation/
│   │   └── validation.py    # M2 Section 4.5 checklist
│   ├── visualization/
│   │   └── plots.py         # 8 publication-quality figures
│   └── api/
│       ├── app.py           # FastAPI application
│       └── schemas.py       # Pydantic request/response models
├── tests/
│   ├── conftest.py          # Synthetic fixtures (no real data needed)
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
└── scripts/
    ├── train.py             # python scripts/train.py
    └── serve.py             # python scripts/serve.py
```

---

## Models implemented

### Quantitative pipeline
- **ARMA-GARCH(1,1) / GJR-GARCH / EGARCH** — BIC model selection (Huang & Luo 2024)
- **Financial Stress Index** — 4-component composite (VIX 30%, GARCH 30%, Drawdown 20%, Credit 20%)
- **Gaussian HMM n=3** — stable / volatile / crisis states (Ang & Timmermann 2012)

### NLP pipeline
- **FinBERT** (ProsusAI/finbert) — GPU FP16, batch inference with checkpointing
- **VADER** — lexicon baseline for comparison (Shobayo et al. 2024)
- **Synthetic VIX proxy** — backward-looking gap-fill, clearly flagged in outputs

### Fusion & explainability
- **Multimodal ensemble** — Logistic Regression + Random Forest + Gradient Boosting
- **SHAP TreeExplainer** — per-crisis feature attribution (Lundberg 2020)
- **Lead-lag analysis** — ±30 trading day cross-correlation with bootstrap CI
- **Granger causality** — sentiment → FSI (Bollen et al. 2011)
