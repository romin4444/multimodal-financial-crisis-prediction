"""
Regression tests for scripts/risk_overlay_run.py.

The risk-overlay script's value depends entirely on three properties holding:

  1. Weights are CAUSAL — the position held over day t+1 must use only
     information available at the close of day t.
  2. Stress z-scores use EXPANDING-window stats (no future mean/std leak).
  3. The bootstrap is REPRODUCIBLE under a fixed seed.

Plus an end-to-end smoke test on synthetic data, which doubles as proof that
the Kaggle-mode synthetic fallback completes cleanly when yfinance is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path so we can import the script module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import risk_overlay_run as ro  # noqa: E402


def _synthetic_spy_vix(seed: int = 0):
    """Deterministic 4-year synthetic SPY + VIX series for unit tests."""
    return ro._synthetic("2018-01-01", "2022-12-31", seed=seed)


class TestCausalWeights:
    """Weights set at close of day t can only depend on data through day t."""

    def test_weight_held_is_shifted_by_one_day(self):
        spy, vix = _synthetic_spy_vix()
        ret, w_held, _ = ro.build_weights(spy, vix, ro.CFG, target_ann_vol=0.12)
        # The first day must have w_held = 0 (no prior info to set a weight).
        assert w_held.iloc[0] == 0.0
        # And w_held must align day-for-day with ret.
        assert w_held.index.equals(ret.index)

    def test_perturbing_future_does_not_change_past_weight(self):
        """The single sharpest causality test: change tomorrow's price, today's
        weight must be byte-identical to the unperturbed case."""
        spy, vix = _synthetic_spy_vix()
        _, w_clean, _ = ro.build_weights(spy, vix, ro.CFG, target_ann_vol=0.12)

        # Pick a day in the middle of the series with a well-defined weight.
        t = 1000
        assert not np.isnan(w_clean.iloc[t])

        # Now corrupt every value from day t+1 onward with extreme moves.
        spy_perturbed = spy.copy()
        vix_perturbed = vix.copy()
        spy_perturbed.iloc[t + 1:] *= 0.5  # 50% crash, repeatedly
        vix_perturbed.iloc[t + 1:] = 80.0  # max-stress VIX everywhere after t

        _, w_perturbed, _ = ro.build_weights(
            spy_perturbed, vix_perturbed, ro.CFG, target_ann_vol=0.12
        )

        # The weight HELD on day t (set at close of t-1 from data through t-1)
        # must be unchanged by any future perturbation.
        assert w_clean.iloc[t] == w_perturbed.iloc[t], (
            f"Causality violated at t={t}: weight changed when future was "
            f"perturbed ({w_clean.iloc[t]} -> {w_perturbed.iloc[t]})"
        )

    def test_weights_are_bounded(self):
        """Vol-targeting + risk-off must respect the [w_min, w_max] hard cap."""
        spy, vix = _synthetic_spy_vix()
        _, w_held, _ = ro.build_weights(spy, vix, ro.CFG, target_ann_vol=0.12)
        assert (w_held >= ro.CFG["w_min"] - 1e-12).all()
        assert (w_held <= ro.CFG["w_max"] + 1e-12).all()


class TestBootstrapReproducibility:
    """The same seed must produce the same CI — otherwise published numbers
    can't be reproduced from the committed artifact."""

    def test_same_seed_same_ci(self):
        spy, vix = _synthetic_spy_vix()
        ret, w_held, _ = ro.build_weights(spy, vix, ro.CFG, target_ann_vol=0.12)
        bt = ro.backtest(ret, w_held, ro.CFG)
        # Smaller B for speed — we're testing reproducibility, not precision.
        cfg = {**ro.CFG, "bootstrap_B": 200}

        ci_a = ro.bootstrap_ci(bt["strat"], bt["bh"], cfg)
        ci_b = ro.bootstrap_ci(bt["strat"], bt["bh"], cfg)

        for key in ("sharpe_diff_ann", "maxdd_reduction"):
            for stat in ("mean", "lo", "hi", "p_le_0"):
                assert ci_a[key][stat] == ci_b[key][stat], (
                    f"Bootstrap {key}.{stat} differs across runs with same seed"
                )

    def test_stationary_bootstrap_idx_is_in_range(self):
        rng = np.random.default_rng(0)
        idx = ro.stationary_bootstrap_idx(n=500, expected_block=21, rng=rng)
        assert len(idx) == 500
        assert idx.min() >= 0 and idx.max() < 500


class TestPerf:
    def test_perf_handles_zero_series(self):
        out = ro.perf(np.zeros(300))
        assert out["sharpe"] == 0
        assert out["maxdd"] == 0

    def test_perf_recovers_known_inputs(self):
        # Constant positive log-return -> known CAGR, ~zero vol, no drawdown.
        r = np.full(252, np.log(1.001))  # ~0.1%/day
        out = ro.perf(r, td=252)
        # np.full().std() is ~3e-18, not strictly 0 — allow numerical noise.
        assert abs(out["vol"]) < 1e-12
        assert out["maxdd"] == 0
        # CAGR should be approximately (1.001)^252 - 1.
        expected_cagr = 1.001 ** 252 - 1
        assert abs(out["cagr"] - expected_cagr) < 1e-6


class TestSmokeEndToEnd:
    """The script must complete cleanly on synthetic data and produce a
    well-formed JSON artifact. Doubles as a CI guard that the Kaggle-mode
    synthetic fallback works."""

    def test_main_runs_on_synthetic(self, monkeypatch, tmp_path):
        # Force the synthetic path: pretend yfinance is unavailable.
        monkeypatch.setattr(ro, "yf", None)
        monkeypatch.setattr(ro, "_resolve_out_dir", lambda: tmp_path)
        # Cut the bootstrap down so the test runs in a few seconds.
        monkeypatch.setattr(ro, "CFG", {**ro.CFG, "bootstrap_B": 50, "end": "2022-12-31"})

        results = ro.main()

        assert results["data_source"].startswith("SYNTHETIC")
        assert results["n_days"] > 1000
        assert "buy_and_hold" in results and "risk_overlay" in results
        assert "bootstrap" in results and "deflated_sharpe" in results
        # §2.5 additions: regime / cost / exposure must round-trip cleanly.
        assert "regime_conditional" in results
        assert "cost_grid" in results and len(results["cost_grid"]) >= 2
        for row in results["cost_grid"]:
            assert {"cost_bps", "strat_maxdd", "maxdd_reduction_pp"} <= row.keys()
        assert "exposure_path" in results

        # JSON artifact must exist and be re-loadable.
        import json
        with open(tmp_path / "risk_overlay_results.json") as fh:
            loaded = json.load(fh)
        assert loaded["selected_target_ann_vol"] in ro.CFG["vol_target_grid"]
        # The figure should also have been written.
        assert (tmp_path / "risk_overlay.png").exists()


class TestEMHConsistency:
    """The script's docstring promises: 'consistent with weak-form EMH,
    the Sharpe edge will likely be small'. Pin a loose upper bound so a
    suspicious future change (e.g. accidentally introducing lookahead)
    produces a noisy failure here."""

    def test_per_period_sharpe_is_modest(self):
        spy, vix = _synthetic_spy_vix()
        ret, w_held, _ = ro.build_weights(spy, vix, ro.CFG, target_ann_vol=0.12)
        bt = ro.backtest(ret, w_held, ro.CFG)
        sharpe_pp = ro._sharpe_per_period(bt["strat"])
        # On synthetic data with our crisis-injected dynamics, a per-period
        # Sharpe above 0.20 (~ 3.2 annualized) would indicate something
        # leaky has crept in. The real-data value is ~0.04.
        assert abs(sharpe_pp) < 0.20, (
            f"per-period Sharpe {sharpe_pp:.3f} is implausibly high — "
            "suspect lookahead leak"
        )
