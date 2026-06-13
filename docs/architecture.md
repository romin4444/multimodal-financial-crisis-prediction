# FCPS architecture

A one-page tour of how data flows through the pipeline. Rendered inline by
GitHub's mermaid support — no external image needed.

## Data → features → models → fusion → evaluation

```mermaid
flowchart LR
    subgraph DATA["Data sources (offline-cached snapshot ships with repo)"]
        SP[S&P 500 OHLCV<br/>yfinance / cache]
        VIX[VIX OHLCV<br/>yfinance / cache]
        FRED[FRED macro<br/>fred_data.csv]
        NEWS[News headlines<br/>NEWS_DATA_DIR + FinBERT]
    end

    subgraph FEAT["Feature engineering (src/features/, src/models/, src/v3/)"]
        ENG[engineer_features<br/>returns, vol, drawdown, momentum]
        GARCH[GARCH / GJR / EGARCH<br/>BIC selection]
        FSI[Financial Stress Index<br/>VIX + GARCH + drawdown + credit<br/>train-only scalers]
        HMM[3-state Gaussian HMM<br/>per-fold refit + filtered posteriors]
        VRP[VIX-orthogonal residuals<br/>Variance Risk Premium]
        SENT[FinBERT / VADER sentiment<br/>+ synthetic VIX proxy]
    end

    subgraph LABEL["Exogenous label (src/v3/labeling.py)"]
        Y[forward 21d max drawdown ≤ -10%]
    end

    subgraph FUSE["Models (scripts/v3_run.py)"]
        LR[Logistic Regression<br/>un-weighted = calibrated]
        RF[RandomForest<br/>n_estimators=300, depth=5]
    end

    subgraph EVAL["Honest evaluation harness (src/v3/)"]
        WF[Purged expanding walk-forward<br/>21d embargo · quarterly refit]
        CPCV[CPCV: 6 groups, k=2]
        METRICS[PR-AUC · Brier skill · ECE<br/>lift · economic backtest]
        DEFL[Deflated Sharpe + PBO]
    end

    subgraph BASE["Baselines you have to beat"]
        BR[base-rate]
        VIXT[VIX-threshold]
        PERS[persistence]
    end

    SP --> ENG
    VIX --> ENG
    FRED --> FSI
    SP --> GARCH
    GARCH --> FSI
    ENG --> FSI
    ENG --> HMM
    FSI --> HMM
    VIX --> VRP
    GARCH --> VRP
    NEWS --> SENT

    ENG --> Y

    ENG --> LR
    HMM --> LR
    SENT --> LR
    VRP --> LR
    ENG --> RF
    HMM --> RF
    SENT --> RF
    VRP --> RF

    LR --> WF
    RF --> WF
    BR --> WF
    VIXT --> WF
    PERS --> WF

    WF --> METRICS
    CPCV --> DEFL
    METRICS --> OUT[outputs/v3_metrics.json]
    DEFL --> OUT
    Y --> WF
```

## Where to look in the code

| Stage | File | What it does |
|---|---|---|
| Market data | `src/data/market.py` | yfinance fetch + cached snapshot |
| Macro data | `src/data/fred.py` | FRED snapshot + daily alignment |
| News (optional) | `src/data/news.py` | NEWS_DATA_DIR → FinBERT |
| Price features | `src/features/engineering.py` | returns, vol, drawdown, momentum |
| GARCH | `src/models/garch.py` | ARMA-GARCH/GJR/EGARCH + BIC selection |
| FSI | `src/models/fsi.py` | train-only scalers, validated vs STLFSI |
| HMM (offline) | `src/models/hmm.py` | n-state selection via BIC |
| HMM (online, per-fold) | `src/v3/online_features.py` | refit + filtered posteriors |
| Sentiment | `src/models/sentiment.py` | FinBERT + VADER + VIX proxy |
| Exogenous label | `src/v3/labeling.py` | forward-drawdown crisis target |
| Walk-forward | `src/v3/walkforward.py` | purged, embargoed, fails loudly on 0 folds |
| CPCV | `src/v3/cpcv.py` | combinatorial purged CV |
| Calibration | `src/v3/calibration.py` | isotonic / sigmoid time-series calibration |
| Baselines | `src/v3/baselines.py` | base-rate / VIX / persistence |
| Metrics | `src/v3/metrics.py` | PR-AUC, Brier skill, ECE, lift, econ backtest |
| Deflated Sharpe | `src/v3/deflated_sharpe.py` | Bailey & López de Prado |
| API | `src/api/app.py` | FastAPI `/predict`, `/predict/batch`, `/metrics`, `/health` |
