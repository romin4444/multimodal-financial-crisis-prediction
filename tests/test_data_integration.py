"""
Integration tests against the committed REAL FRED snapshot (fred_data.csv).

Synthetic tests can't catch date-alignment, frequency-mismatch, or NaN-pattern
bugs that only appear with real economic data. These tests load the actual
committed snapshot and exercise the alignment + macro-feature join on it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FRED_SNAPSHOT = REPO_ROOT / "fred_data.csv"


@pytest.fixture(scope="module")
def real_fred() -> pd.DataFrame:
    if not FRED_SNAPSHOT.exists():
        pytest.skip("committed fred_data.csv snapshot not present")
    return pd.read_csv(FRED_SNAPSHOT, index_col=0, parse_dates=True)


class TestRealFredAlignment:
    def test_aligns_exactly_to_trading_calendar(self, real_fred):
        from src.data.fred import align_fred_to_trading_days
        idx = pd.bdate_range("2005-01-03", "2019-12-31")
        daily = align_fred_to_trading_days(real_fred, idx)
        # Real-world alignment must produce exactly the requested trading index
        assert daily.index.equals(idx)

    def test_well_populated_series_present(self, real_fred):
        from src.data.fred import align_fred_to_trading_days
        idx = pd.bdate_range("2005-01-03", "2019-12-31")
        daily = align_fred_to_trading_days(real_fred, idx)
        # The yield-curve slope is available for the full modern period
        assert "yield_spread" in daily.columns
        assert daily["yield_spread"].notna().mean() > 0.8

    def test_ffill_limit_caps_propagation(self):
        """A single observation must not propagate beyond cfg.data.fred_ffill_limit
        trading days — this is the leakage guard against stale macro values."""
        from src.config import cfg
        from src.data.fred import align_fred_to_trading_days
        idx = pd.bdate_range("2010-01-01", periods=400)
        # 5 sparse observations spaced 40 trading days apart (gaps > the cap)
        vals = [np.nan] * len(idx)
        for i in range(5):
            vals[i * 40] = float(i)
        sparse = pd.DataFrame({"x": vals}, index=idx)
        out = align_fred_to_trading_days(sparse, idx)
        # Each value fills forward at most `limit` days, so 40-day gaps leave holes
        assert out["x"].notna().mean() < 1.0
        # And never more than (limit + 1) filled days per observation
        assert out["x"].notna().sum() <= 5 * (cfg.data.fred_ffill_limit + 1)


class TestRealFredMacro:
    def test_macro_features_build_on_real_fred(self, real_fred):
        from src.v3.macro_features import build_macro_features, usable_macro_cols
        idx = pd.bdate_range("2005-01-03", "2019-12-31")
        rng = np.random.default_rng(0)
        market = {}
        for k in ["sp500", "vix", "aapl", "jpm", "xom", "gs"]:
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
            market[k] = pd.DataFrame(
                {"Close": c, "Volume": rng.integers(1_000_000, 5_000_000, len(idx))}, index=idx
            )
        macro = build_macro_features(real_fred, market, idx)
        assert macro.index.equals(idx)
        # Yield slope is a real, well-populated VIX-orthogonal feature
        assert "yield_slope" in macro.columns
        assert macro["yield_slope"].notna().mean() > 0.8
        # Coverage filter returns at least the rate/curve features for real data
        usable = usable_macro_cols(macro)
        assert "yield_slope" in usable
