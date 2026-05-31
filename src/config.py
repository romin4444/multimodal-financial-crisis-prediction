"""
Centralised configuration loading with Pydantic validation.
Single source of truth for all parameters — loaded once at import time.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


# ─── Sub-models ──────────────────────────────────────────────────────────────

class PathsConfig(BaseModel):
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("outputs/models")
    log_dir: Path = Path("logs")

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.output_dir, self.model_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    start_date: str = "1990-01-01"
    end_date: str = "2024-12-31"
    index_ticker: str = "^GSPC"
    vix_ticker: str = "^VIX"
    stock_tickers: List[str] = ["AAPL", "JPM", "XOM", "GS"]
    fred_series: Dict[str, str] = {}
    fred_ffill_limit: int = 22
    max_headlines: int = 400_000
    news_min_headline_len: int = 15
    yfinance_retries: int = 3
    yfinance_retry_backoff: int = 2


class FSIConfig(BaseModel):
    weights: Dict[str, float] = {"vix": 0.30, "garch": 0.30, "drawdown": 0.20, "credit": 0.20}
    correlation_target: float = 0.60

    @field_validator("weights")
    @classmethod
    def weights_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"FSI weights must sum to 1.0, got {total:.4f}")
        return v


class GARCHSpecConfig(BaseModel):
    vol: str
    p: int
    o: int
    q: int
    label: str


class GARCHConfig(BaseModel):
    specs: List[GARCHSpecConfig] = []
    max_iter: int = 2000
    ftol: float = 1e-9


class HMMConfig(BaseModel):
    n_states_candidates: List[int] = [2, 3, 4]
    canonical_n_states: int = 3
    n_init_seeds: int = 50
    n_iter: int = 200
    tol: float = 1e-5
    features: List[str] = ["log_ret", "garch_var", "vol_21d", "FSI"]


class FinBERTConfig(BaseModel):
    model_name: str = "ProsusAI/finbert"
    batch_size_gpu: int = 128
    batch_size_cpu: int = 32
    max_length: int = 128
    panic_threshold: float = 0.40
    checkpoint_every_n: int = 5000


class LeadLagConfig(BaseModel):
    max_lag_days: int = 30
    bootstrap_n: int = 1000
    crisis_window_bootstrap_n: int = 200


class ModelHParams(BaseModel):
    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 2000


class RFHParams(BaseModel):
    n_estimators: int = 500
    max_depth: int = 6
    min_samples_leaf: int = 10
    n_jobs: int = -1


class GBHParams(BaseModel):
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.8


class FusionConfig(BaseModel):
    prediction_horizon_days: int = 5
    f1_target: float = 0.70
    feature_columns: List[str] = []
    logistic_regression: ModelHParams = ModelHParams()
    random_forest: RFHParams = RFHParams()
    gradient_boosting: GBHParams = GBHParams()


class CrisisWindow(BaseModel):
    start: str
    end: str

    def as_tuple(self) -> Tuple[str, str]:
        return (self.start, self.end)


class VisualizationConfig(BaseModel):
    dpi: int = 200
    palette: Dict[str, str] = {}


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    min_history_days: int = 126


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    rotation_mb: int = 100
    retention_days: int = 30


# ─── Root config ─────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    seed: int = 42
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    fsi: FSIConfig = FSIConfig()
    garch: GARCHConfig = GARCHConfig()
    hmm: HMMConfig = HMMConfig()
    finbert: FinBERTConfig = FinBERTConfig()
    lead_lag: LeadLagConfig = LeadLagConfig()
    fusion: FusionConfig = FusionConfig()
    crisis_windows: Dict[str, CrisisWindow] = {}
    nber_recessions: List[List[str]] = []
    visualization: VisualizationConfig = VisualizationConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()

    @property
    def crisis_window_tuples(self) -> Dict[str, Tuple[str, str]]:
        return {name: w.as_tuple() for name, w in self.crisis_windows.items()}


# ─── Loader ──────────────────────────────────────────────────────────────────

_CONFIG_SEARCH_PATHS = [
    Path("config.yaml"),
    Path(__file__).parent.parent / "config.yaml",
]


def load_config(path: Optional[Path] = None) -> AppConfig:
    """
    Load and validate configuration from YAML file.
    Environment variable CONFIG_PATH overrides the default search.
    """
    if path is None:
        env_path = os.environ.get("CONFIG_PATH")
        if env_path:
            path = Path(env_path)
        else:
            for candidate in _CONFIG_SEARCH_PATHS:
                if candidate.exists():
                    path = candidate
                    break

    if path is None or not path.exists():
        # Return defaults — useful for tests
        return AppConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = AppConfig.model_validate(raw or {})
    cfg.paths.ensure_dirs()
    return cfg


# Module-level singleton — import this everywhere
cfg: AppConfig = load_config()
