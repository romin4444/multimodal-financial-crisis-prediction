"""
pytest fixtures — synthetic data shared across all test modules.
No real API calls; no real model downloads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def trading_index() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", "2023-12-31", freq="B")


@pytest.fixture(scope="session")
def synthetic_prices(trading_index) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with realistic log-normal price dynamics."""
    rng = np.random.default_rng(42)
    n = len(trading_index)
    log_ret = rng.normal(0.0003, 0.01, size=n)
    price = 1000 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(
        {
            "Open": price * (1 + rng.uniform(-0.002, 0.002, n)),
            "High": price * (1 + rng.uniform(0.001, 0.005, n)),
            "Low": price * (1 - rng.uniform(0.001, 0.005, n)),
            "Close": price,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=trading_index,
    )


@pytest.fixture(scope="session")
def synthetic_vix(trading_index) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = len(trading_index)
    vix = 15 + 5 * np.sin(np.linspace(0, 6 * np.pi, n)) + rng.normal(0, 2, n)
    vix = np.clip(vix, 9, 80)
    return pd.DataFrame({"Close": vix}, index=trading_index)


@pytest.fixture(scope="session")
def feat_df(synthetic_prices, synthetic_vix):
    from src.features.engineering import engineer_features
    df = engineer_features(synthetic_prices, synthetic_vix)
    # Drop garch_var placeholder column so .dropna() works as expected
    return df.drop(columns=["garch_var"]).dropna()


@pytest.fixture(scope="session")
def synthetic_news(trading_index) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    headlines = [
        "Markets fall sharply amid economic concerns",
        "Stock rally extends for third week",
        "Investors brace for volatility as earnings season begins",
        "Federal Reserve signals rate hike concerns",
        "Tech sector leads broad market advance",
    ]
    n = len(trading_index)
    chosen = rng.choice(headlines, size=n)
    return pd.DataFrame({"date": trading_index, "headline": chosen})


@pytest.fixture(scope="session")
def synthetic_sentiment(trading_index) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    n = len(trading_index)
    fear = np.clip(rng.beta(2, 5, n), 0, 1)
    df = pd.DataFrame(
        {
            "fear_index": fear,
            "panic_signal": (fear > 0.4).astype(int),
            "fear_3d": pd.Series(fear).rolling(3, min_periods=1).mean().values,
            "fear_7d": pd.Series(fear).rolling(7, min_periods=1).mean().values,
            "fear_21d": pd.Series(fear).rolling(21, min_periods=1).mean().values,
            "headline_count": rng.integers(0, 50, n),
        },
        index=trading_index,
    )
    return df


@pytest.fixture(scope="session")
def synthetic_regime(trading_index) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = len(trading_index)
    regimes = rng.choice([0, 1, 2], size=n, p=[0.60, 0.30, 0.10])
    probs = rng.dirichlet([3, 1.5, 0.5], size=n)
    return pd.DataFrame(
        {
            "regime": regimes,
            "prob_stable": probs[:, 0],
            "prob_volatile": probs[:, 1],
            "prob_crisis": probs[:, 2],
        },
        index=trading_index,
    )
