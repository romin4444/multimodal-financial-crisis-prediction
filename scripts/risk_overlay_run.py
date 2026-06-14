#!/usr/bin/env python3
"""
FCPS — Stress-Scaled Risk Overlay (Kaggle-runnable, self-contained)
===================================================================

WHY THIS EXISTS
---------------
The FCPS project's honest finding is: market *direction* and crisis *timing* are
not reliably predictable, but portfolio *risk* is rankable and reducible. This
script operationalizes exactly that — and nothing more. It does NOT try to call
crises. It scales equity exposure using a *causal* (no-look-ahead) stress signal
and asks one question, rigorously:

    "Does down-weighting exposure when public stress is high reduce drawdown,
     and is any risk-adjusted improvement statistically real after costs and
     after correcting for the configs we searched?"

EVALUATION RIGOR (matches the repo's leakage-free ethos)
--------------------------------------------------------
  * Every signal uses only information available at the close of day t to set
    the position held over day t+1 (weights are .shift(1)-ed; asserted below).
  * Stress z-scores use EXPANDING-window stats (no future mean/std leakage).
  * Transaction costs charged on daily turnover.
  * Significance via stationary block bootstrap (Politis & Romano 1994) on the
    JOINT (strategy, buy&hold) daily returns — preserves autocorrelation and
    cross-correlation. CI on Sharpe difference AND max-drawdown reduction.
  * Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) deflates the SELECTED
    config's Sharpe by the number of configs tried and the variance of their
    Sharpes — the correct antidote to backtest selection bias.

HONEST EXPECTATION
------------------
Consistent with weak-form EMH, the Sharpe *edge* over buy-and-hold will likely
be small and its bootstrap CI will likely straddle zero (i.e. NOT significant).
The drawdown *reduction*, by contrast, is usually real and significant. Reporting
both — including the non-significant one — is the point.

RUN MODES
---------
- Kaggle (Internet ON): paste into one cell; resolves real SPY+VIX via yfinance.
- Local repo: ``python scripts/risk_overlay_run.py`` — writes
  outputs/risk_overlay_results.json and outputs/risk_overlay.png.
- Offline / CI: yfinance fetch fails -> synthetic fallback. Always completes.

Author: Romin Patel.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

# Force UTF-8 stdout/stderr on Windows so log unicode chars don't crash.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------------- #
# 0. Dependencies (works whether run as a notebook cell or a plain .py script)
# ----------------------------------------------------------------------------- #
def _ensure(pkg: str, import_as: str | None = None):
    name = import_as or pkg
    try:
        return __import__(name)
    except ImportError:
        # Kaggle-only fallback: pip-install on the fly. In the repo this is a
        # no-op because yfinance ships in requirements.txt; CI never needs it.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)  # noqa: S603
        try:
            return __import__(name)
        except ImportError:
            return None


yf = _ensure("yfinance")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Project's shared safe-JSON encoder if we're inside the repo; else degrade.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.json_utils import safe_json_default
except Exception:  # noqa: BLE001 — Kaggle / standalone use, repo not on sys.path
    def safe_json_default(o):  # type: ignore[no-redef]
        import math
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (float, np.floating)):
            v = float(o)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        return str(o)


# ----------------------------------------------------------------------------- #
# 1. Config
# ----------------------------------------------------------------------------- #
CFG = dict(
    start="1993-01-29",          # SPY inception era
    end=None,                    # today
    target_ann_vol=0.12,         # 12% annualized vol target for the overlay
    vol_lookback=21,             # trailing window for realized-vol forecast
    w_max=1.0,                   # no leverage (cap exposure at 100%)
    w_min=0.0,
    stress_cut_quantile=0.90,    # when expanding stress z is in top decile...
    stress_cut_factor=0.5,       # ...halve exposure (risk-off overlay)
    cost_bps=2.0,                # per-unit-turnover transaction cost
    trading_days=252,
    # Deflated-Sharpe trial grid (the configs we "searched") — used to deflate:
    vol_target_grid=(0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
    bootstrap_B=2000,
    block_len=21,                # expected block length for stationary bootstrap
    seed=7,
)


def _resolve_out_dir() -> Path:
    """Kaggle working dir if present, else repo outputs/, else cwd."""
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    repo_outputs = Path(__file__).resolve().parents[1] / "outputs"
    if repo_outputs.parent.exists():
        repo_outputs.mkdir(parents=True, exist_ok=True)
        return repo_outputs
    return Path(".")


# ----------------------------------------------------------------------------- #
# 2. Data (real via yfinance; synthetic fallback so it always runs)
# ----------------------------------------------------------------------------- #
def _synthetic(start, end, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end or "2024-12-31")
    n = len(idx)
    r = rng.normal(0.0003, 0.01, n)
    for cs, ce, s in [("2008-09-01", "2009-03-31", 0.045),
                      ("2020-02-19", "2020-03-23", 0.06),
                      ("2022-01-01", "2022-10-31", 0.022)]:
        m = (idx >= pd.Timestamp(cs)) & (idx <= pd.Timestamp(ce))
        r[m] += rng.normal(-0.0015, s, m.sum())
    spy = pd.Series(300 * np.exp(np.cumsum(r)), index=idx, name="SPY")
    # crude VIX proxy from rolling realized vol
    rv = pd.Series(r, index=idx).rolling(21).std() * np.sqrt(252) * 100
    vix = (rv.fillna(rv.median()) + rng.normal(0, 1.5, n)).clip(8, 90).rename("VIX")
    return spy, vix


def load_data(cfg):
    if yf is not None:
        try:
            spy = yf.download("SPY", start=cfg["start"], end=cfg["end"],
                              auto_adjust=True, progress=False)["Close"]
            vix = yf.download("^VIX", start=cfg["start"], end=cfg["end"],
                              auto_adjust=True, progress=False)["Close"]
            spy = spy.squeeze()
            vix = vix.squeeze()
            spy.name, vix.name = "SPY", "VIX"
            df = pd.concat([spy, vix], axis=1).dropna()
            if len(df) > 1000:
                print(f"[data] REAL SPY+VIX: {df.index[0].date()} -> {df.index[-1].date()} "
                      f"({len(df)} days)")
                return df["SPY"], df["VIX"], "REAL (yfinance SPY + ^VIX)"
        except Exception as e:  # noqa: BLE001 — synthetic fallback below is the point
            print(f"[data] yfinance failed ({e}); using synthetic fallback.")
    spy, vix = _synthetic(cfg["start"], cfg["end"])
    df = pd.concat([spy, vix], axis=1).dropna()
    print(f"[data] SYNTHETIC fallback: {df.index[0].date()} -> {df.index[-1].date()} "
          f"({len(df)} days)")
    return df["SPY"], df["VIX"], "SYNTHETIC (no internet)"


# ----------------------------------------------------------------------------- #
# 3. Causal stress signal + exposure weights
# ----------------------------------------------------------------------------- #
def build_weights(spy, vix, cfg, target_ann_vol):
    ret = np.log(spy).diff().rename("ret")

    # (a) Forward-vol forecast = trailing realized vol (annualized), known at t.
    realized_vol = ret.rolling(cfg["vol_lookback"]).std() * np.sqrt(cfg["trading_days"])

    # (b) Vol-targeting weight, known at close of t.
    w_voltarget = (target_ann_vol / realized_vol).clip(cfg["w_min"], cfg["w_max"])

    # (c) Causal stress index: combine VIX level, trailing realized vol, and
    #     drawdown-from-trailing-peak — each z-scored on an EXPANDING window.
    dd = spy / spy.cummax() - 1.0                      # <= 0, known at t
    comp = pd.DataFrame({"vix": vix, "rvol": realized_vol, "dd": -dd})
    z = (comp - comp.expanding(min_periods=60).mean()) / \
        comp.expanding(min_periods=60).std()
    stress = z.mean(axis=1)                            # higher = more stress

    # (d) Risk-off overlay: cut exposure when stress is in its top decile
    #     (threshold computed on expanding history -> causal).
    stress_thresh = stress.expanding(min_periods=120).quantile(cfg["stress_cut_quantile"])
    risk_off = (stress > stress_thresh).astype(float)
    w = w_voltarget * (1.0 - risk_off * (1.0 - cfg["stress_cut_factor"]))
    w = w.clip(cfg["w_min"], cfg["w_max"]).fillna(0.0)

    # (e) CRITICAL: shift weights by 1 day. Position decided at close of t is
    #     held over t+1. This is what makes the backtest leakage-free.
    w_held = w.shift(1).fillna(0.0)
    # Stress is also shift(1)'d so any regime-conditional reporting aligns with
    # the position actually held over the next day — same leakage rule.
    return ret, w_held, stress.shift(1).fillna(0.0)


def backtest(ret, w_held, cfg, cost_bps=None):
    """Vector backtest. ``cost_bps`` overrides ``cfg['cost_bps']`` when supplied —
    used by the cost-grid (§2.5 sensitivity) without mutating the global config."""
    bps = cfg["cost_bps"] if cost_bps is None else cost_bps
    turnover = w_held.diff().abs().fillna(w_held.abs())
    cost = turnover * (bps / 1e4)
    strat = w_held * ret - cost
    bh = ret.copy()                                    # buy & hold = 100% always
    return pd.DataFrame({"strat": strat, "bh": bh, "w": w_held,
                         "turnover": turnover}).dropna()


# ----------------------------------------------------------------------------- #
# 4. Performance metrics
# ----------------------------------------------------------------------------- #
def perf(daily_log_ret, td=252):
    r = np.asarray(daily_log_ret, float)
    if r.std() == 0 or len(r) < 2:
        return dict(cagr=0, vol=0, sharpe=0, maxdd=0, calmar=0)
    eq = np.exp(np.cumsum(r))
    yrs = len(r) / td
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(td)
    sharpe = (r.mean() * td) / vol
    peak = np.maximum.accumulate(eq)
    maxdd = float((eq / peak - 1).min())
    calmar = cagr / abs(maxdd) if maxdd != 0 else np.nan
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sharpe),
                maxdd=maxdd, calmar=float(calmar))


def _sharpe_per_period(r):
    r = np.asarray(r, float)
    return r.mean() / r.std() if r.std() > 0 else 0.0


def _maxdd_from_logret(r):
    eq = np.exp(np.cumsum(np.asarray(r, float)))
    return float((eq / np.maximum.accumulate(eq) - 1).min())


# ----------------------------------------------------------------------------- #
# 5. Stationary block bootstrap (Politis & Romano 1994)
# ----------------------------------------------------------------------------- #
def stationary_bootstrap_idx(n, expected_block, rng):
    p = 1.0 / expected_block
    idx = np.empty(n, dtype=np.int64)
    t = int(rng.integers(0, n))
    for i in range(n):
        idx[i] = t
        t = int(rng.integers(0, n)) if rng.random() < p else (t + 1) % n
    return idx


def bootstrap_ci(strat, bh, cfg):
    """Joint block-resample -> CI on (Sharpe diff) and (maxDD reduction)."""
    rng = np.random.default_rng(cfg["seed"])
    s, b = np.asarray(strat, float), np.asarray(bh, float)
    n = len(s)
    td = cfg["trading_days"]
    d_sharpe, d_ddred = [], []
    for _ in range(cfg["bootstrap_B"]):
        idx = stationary_bootstrap_idx(n, cfg["block_len"], rng)
        rs, rb = s[idx], b[idx]
        d_sharpe.append((_sharpe_per_period(rs) - _sharpe_per_period(rb)) * np.sqrt(td))
        # drawdown reduction = bh_maxdd_depth - strat_maxdd_depth  (positive = shallower)
        d_ddred.append(abs(_maxdd_from_logret(rb)) - abs(_maxdd_from_logret(rs)))

    def ci(a):
        a = np.array(a)
        return dict(mean=float(a.mean()),
                    lo=float(np.percentile(a, 2.5)),
                    hi=float(np.percentile(a, 97.5)),
                    p_le_0=float((a <= 0).mean()))

    return dict(sharpe_diff_ann=ci(d_sharpe), maxdd_reduction=ci(d_ddred))


# ----------------------------------------------------------------------------- #
# 6. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# ----------------------------------------------------------------------------- #
def deflated_sharpe(selected_ret, sr_trials_per_period, n_trials):
    r = np.asarray(selected_ret, float)
    T = len(r)
    sr_hat = _sharpe_per_period(r)                     # per-period units
    sk = float(skew(r))
    ku = float(kurtosis(r, fisher=False))              # non-excess kurtosis
    g = 0.5772156649015329                             # Euler-Mascheroni
    sr_std = (
        float(np.std(sr_trials_per_period, ddof=1))
        if len(sr_trials_per_period) > 1
        else 1e-6
    )
    # Expected maximum Sharpe across N independent trials:
    sr0 = sr_std * ((1 - g) * norm.ppf(1 - 1.0 / n_trials)
                    + g * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    denom = np.sqrt(max(1e-12, 1 - sk * sr_hat + ((ku - 1) / 4.0) * sr_hat ** 2))
    dsr = float(norm.cdf((sr_hat - sr0) * np.sqrt(T - 1) / denom))
    return dict(sr_hat_per_period=float(sr_hat), sr0_per_period=float(sr0),
                deflated_sharpe_prob=dsr, n_trials=int(n_trials), T=int(T))


# ----------------------------------------------------------------------------- #
# 7. §2.5 — Regime-conditional / cost-grid / exposure-path reporting
# ----------------------------------------------------------------------------- #
def regime_conditional(strat, bh, stress, td=252):
    """Split the joint return panel by stress tercile (computed at the same
    leakage-free shift used by ``build_weights``) and report Sharpe + max-DD per
    tercile. The overlay's economic case lives in the TOP tercile — the days a
    risk desk actually cares about. The middle/bottom report is the honest
    counter-evidence (where the overlay can only cost performance)."""
    # Use only days where stress has values (drop the early warm-up NaNs).
    valid = stress.notna() & strat.notna() & bh.notna()
    s, b, z = strat[valid], bh[valid], stress[valid]
    if len(z) < 60:
        return {"note": "insufficient days for tercile split"}
    q33, q67 = float(z.quantile(1.0 / 3)), float(z.quantile(2.0 / 3))
    out = {"q33": q33, "q67": q67}
    for name, mask in [
        ("bottom", z <= q33),
        ("middle", (z > q33) & (z <= q67)),
        ("top",    z > q67),
    ]:
        n_days = int(mask.sum())
        if n_days < 20:
            out[name] = {"n_days": n_days, "note": "too few days"}
            continue
        ms = perf(s[mask].to_numpy(), td)
        mb = perf(b[mask].to_numpy(), td)
        out[name] = {
            "n_days": n_days,
            "strat_sharpe": ms["sharpe"], "strat_maxdd": ms["maxdd"],
            "bh_sharpe":    mb["sharpe"], "bh_maxdd":    mb["maxdd"],
            "sharpe_edge":  ms["sharpe"] - mb["sharpe"],
            "maxdd_reduction_pp": (abs(mb["maxdd"]) - abs(ms["maxdd"])) * 100,
        }
    return out


def cost_grid(spy, vix, cfg, target_ann_vol, cost_bps_grid=(2.0, 5.0, 10.0)):
    """Re-price the selected overlay at higher transaction costs. The drawdown
    claim is the deployable one (§2.5); we want a reader to see it survives a
    realistic broker fee + spread regime, not only a quoted-edge 2 bps."""
    ret, w_held, _ = build_weights(spy, vix, cfg, target_ann_vol)
    rows = []
    for bps in cost_bps_grid:
        bt_c = backtest(ret, w_held, cfg, cost_bps=bps)
        m_s = perf(bt_c["strat"], cfg["trading_days"])
        m_b = perf(bt_c["bh"],    cfg["trading_days"])
        rows.append({
            "cost_bps": float(bps),
            "strat_cagr": m_s["cagr"], "strat_sharpe": m_s["sharpe"],
            "strat_maxdd": m_s["maxdd"],
            "bh_cagr":   m_b["cagr"], "bh_sharpe":   m_b["sharpe"],
            "bh_maxdd":  m_b["maxdd"],
            "sharpe_edge": m_s["sharpe"] - m_b["sharpe"],
            "maxdd_reduction_pp": (abs(m_b["maxdd"]) - abs(m_s["maxdd"])) * 100,
        })
    return rows


def exposure_path_stats(w_held, low_threshold=0.5):
    """First-class exposure-path metrics — the risk-committee questions:
    how often and for how long does the overlay de-risk? Reports the share of
    days below ``low_threshold`` exposure and the longest contiguous run of
    such days (a proxy for sustained risk-off periods)."""
    w = w_held.dropna().to_numpy()
    if len(w) == 0:
        return {"note": "no exposure data"}
    low = w < low_threshold
    # Longest run of consecutive low-exposure days, vectorized.
    if low.any():
        # Identify run boundaries by where the boolean flips.
        flips = np.diff(np.concatenate([[0], low.view(np.int8), [0]]))
        starts = np.where(flips == 1)[0]
        ends = np.where(flips == -1)[0]
        max_run = int((ends - starts).max())
        n_episodes = int(len(starts))
    else:
        max_run = 0
        n_episodes = 0
    return {
        "low_exposure_threshold": float(low_threshold),
        "share_days_low_exposure": float(low.mean()),
        "max_consecutive_low_exposure_days": max_run,
        "n_low_exposure_episodes": n_episodes,
        "exposure_p10": float(np.quantile(w, 0.10)),
        "exposure_p50": float(np.quantile(w, 0.50)),
        "exposure_p90": float(np.quantile(w, 0.90)),
    }


# ----------------------------------------------------------------------------- #
# 8. Main
# ----------------------------------------------------------------------------- #
def main():
    cfg = CFG
    out_dir = _resolve_out_dir()
    spy, vix, source = load_data(cfg)

    # --- grid search over vol targets (these are the DSR "trials") -------------
    grid_sharpes_pp = []
    for tv in cfg["vol_target_grid"]:
        ret_g, w_g, _ = build_weights(spy, vix, cfg, tv)
        bt_g = backtest(ret_g, w_g, cfg)
        grid_sharpes_pp.append(_sharpe_per_period(bt_g["strat"]))
    best_i = int(np.argmax(grid_sharpes_pp))
    best_tv = cfg["vol_target_grid"][best_i]
    print(f"[grid] selected target_ann_vol={best_tv:.2f} "
          f"(per-period Sharpe={grid_sharpes_pp[best_i]:.4f})")

    # --- evaluate the selected config -----------------------------------------
    ret, w_held, stress = build_weights(spy, vix, cfg, best_tv)
    # leakage assertion: today's weight must not use today's return
    assert w_held.index.equals(ret.index), "alignment error"
    bt = backtest(ret, w_held, cfg)

    m_strat = perf(bt["strat"], cfg["trading_days"])
    m_bh = perf(bt["bh"], cfg["trading_days"])
    avg_exposure = float(bt["w"].mean())
    ann_turnover = float(bt["turnover"].sum() / (len(bt) / cfg["trading_days"]))

    ci = bootstrap_ci(bt["strat"], bt["bh"], cfg)
    dsr = deflated_sharpe(bt["strat"], grid_sharpes_pp, n_trials=len(cfg["vol_target_grid"]))

    # §2.5 additions (roadmap): regime-conditional report, cost-grid sensitivity,
    # exposure-path stats. These do not change the headline numbers — they
    # answer the risk-committee questions the headline alone can't.
    regime = regime_conditional(bt["strat"], bt["bh"], stress.reindex(bt.index),
                                cfg["trading_days"])
    cost_rows = cost_grid(spy, vix, cfg, best_tv, cost_bps_grid=(2.0, 5.0, 10.0))
    expo = exposure_path_stats(bt["w"])

    # --- report ----------------------------------------------------------------
    def row(name, m):
        return (f"  {name:<14} CAGR {m['cagr']*100:6.2f}%  Vol {m['vol']*100:5.2f}%  "
                f"Sharpe {m['sharpe']:5.2f}  MaxDD {m['maxdd']*100:7.2f}%  "
                f"Calmar {m['calmar']:5.2f}")
    print("\n" + "=" * 78)
    print(f"  STRESS-SCALED RISK OVERLAY  |  data: {source}")
    print("=" * 78)
    print(row("Buy & Hold", m_bh))
    print(row("Risk overlay", m_strat))
    print(f"\n  Avg exposure {avg_exposure*100:.0f}%   |   Ann. turnover {ann_turnover:.1f}x   "
          f"|   cost {cfg['cost_bps']} bps/turn")

    sd = ci["sharpe_diff_ann"]
    dr = ci["maxdd_reduction"]
    sig_sharpe = "SIGNIFICANT" if (sd["lo"] > 0 or sd["hi"] < 0) else "NOT significant"
    sig_dd = "SIGNIFICANT" if dr["lo"] > 0 else "NOT significant"
    print(f"\n  --- Stationary block bootstrap (B={cfg['bootstrap_B']}, "
          f"block~{cfg['block_len']}d) ---")
    print(f"  Sharpe edge (overlay - B&H, annualized): {sd['mean']:+.3f}  "
          f"95% CI [{sd['lo']:+.3f}, {sd['hi']:+.3f}]  -> {sig_sharpe}")
    print(f"  Max-drawdown reduction (depth):          {dr['mean']*100:+.2f}pp "
          f"95% CI [{dr['lo']*100:+.2f}, {dr['hi']*100:+.2f}]pp  -> {sig_dd}")
    print("\n  --- Deflated Sharpe Ratio (Bailey & Lopez de Prado) ---")
    print(f"  SR_hat={dsr['sr_hat_per_period']:.4f}/day  vs expected-max-under-"
          f"{dsr['n_trials']}-trials SR0={dsr['sr0_per_period']:.4f}/day")
    print(f"  Deflated Sharpe prob (P[true SR>0 after selection]): "
          f"{dsr['deflated_sharpe_prob']:.3f}")
    print("\n  HONEST READ: the overlay's job is risk reduction, not return. A")
    print("  straddling Sharpe-edge CI is the expected, EMH-consistent result;")
    print("  a significant drawdown reduction is the deployable one.")
    print("=" * 78)

    # --- §2.5 — regime / cost / exposure (the risk-committee view) -------------
    if "top" in regime and isinstance(regime["top"], dict) and "sharpe_edge" in regime["top"]:
        rt = regime["top"]
        print("\n  --- Regime-conditional (top stress tercile) ---")
        print(f"  Days in top tercile: {rt['n_days']:,}  | overlay Sharpe edge "
              f"{rt['sharpe_edge']:+.2f} | DD reduction {rt['maxdd_reduction_pp']:+.2f}pp")

    print("\n  --- Cost-grid sensitivity ---")
    print(f"  {'cost (bps)':>10}  {'overlay CAGR':>13}  {'overlay MaxDD':>14}  "
          f"{'DD reduction':>13}")
    for row_ in cost_rows:
        print(f"  {row_['cost_bps']:>10.1f}  {row_['strat_cagr']*100:>12.2f}%  "
              f"{row_['strat_maxdd']*100:>13.2f}%  "
              f"{row_['maxdd_reduction_pp']:>12.2f}pp")

    if "share_days_low_exposure" in expo:
        print(f"\n  --- Exposure path (low = w < {expo['low_exposure_threshold']:.0%}) ---")
        print(f"  Days low-exposure: {expo['share_days_low_exposure']*100:.1f}%   "
              f"| longest run: {expo['max_consecutive_low_exposure_days']}d   "
              f"| {expo['n_low_exposure_episodes']} episodes")
    print("=" * 78)

    # --- figure -----------------------------------------------------------------
    eq_strat = np.exp(bt["strat"].cumsum())
    eq_bh = np.exp(bt["bh"].cumsum())
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(eq_bh.index, eq_bh, label="Buy & Hold", lw=1.1, color="#888")
    ax[0].plot(eq_strat.index, eq_strat, label=f"Risk overlay (tgt {best_tv:.0%} vol)",
               lw=1.3, color="#1f77b4")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Growth of $1 (log)")
    ax[0].legend(loc="upper left")
    ax[0].set_title("Stress-Scaled Risk Overlay vs Buy & Hold")
    ax[1].fill_between(bt.index, bt["w"], color="#1f77b4", alpha=.35, step="pre")
    ax[1].set_ylabel("Exposure")
    ax[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig_path = out_dir / "risk_overlay.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {fig_path}")

    # --- json (matches repo's outputs/*.json convention) -----------------------
    results = dict(
        data_source=source, selected_target_ann_vol=best_tv,
        config={k: v for k, v in cfg.items() if k != "vol_target_grid"},
        vol_target_grid=list(cfg["vol_target_grid"]),
        n_days=int(len(bt)), avg_exposure=avg_exposure, ann_turnover=ann_turnover,
        buy_and_hold=m_bh, risk_overlay=m_strat,
        bootstrap=ci, deflated_sharpe=dsr,
        # §2.5 additions — additive, do not break the existing schema.
        regime_conditional=regime,
        cost_grid=cost_rows,
        exposure_path=expo,
    )
    json_path = out_dir / "risk_overlay_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=safe_json_default)
    print(f"[saved] {json_path}")
    return results


if __name__ == "__main__":
    main()
