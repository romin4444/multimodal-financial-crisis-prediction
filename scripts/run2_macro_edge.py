#!/usr/bin/env python3
"""
Run 2 — the real test of the #1 requirement: can VIX-ORTHOGONAL macro signal
(full-history credit spread + yield curve + funding + oil) beat a VIX threshold
out-of-sample, under the honest CPCV + PBO + Deflated-Sharpe gates?

Uses the REAL FRED key (from .env / FRED_API_KEY). Long-history credit proxy is
BAA10Y (Moody's Baa - 10Y, daily since 1986) because the ICE HY OAS series is
relicensed to 2023+ on FRED. Built on the tested src/v3 modules.

Author: Romin Patel.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── tested project modules ───────────────────────────────────────────────────
from src.v3.cpcv import CPCVConfig, probability_of_backtest_overfitting, run_cpcv
from src.v3.deflated_sharpe import deflated_sharpe_ratio
from src.v3.metrics import classification_metrics
from src.v3.calibration import make_calibrated_factory
from src.v3.causal_regime import filtered_state_proba
from src.v3.labeling import crisis_label

HORIZONS = [10, 21, 63]
THRESHOLDS = [0.07, 0.10]
LONG_HISTORY_MACRO = {       # series_id -> column ; all daily/weekly back to 1990
    "BAA10Y": "credit",      # Moody's Baa - 10Y (credit-spread proxy, VIX-orthogonal)
    "T10Y2Y": "slope2y",     # yield-curve slope
    "T10Y3M": "slope3m",     # yield-curve slope (recession signal)
    "DGS3MO": "tbill3m",     # short rate (funding)
    "STLFSI4": "stlfsi",     # St Louis stress (positive-control only)
}


def _load_env():
    p = Path(__file__).parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def fetch_macro(key: str, start="1990-01-01", end="2024-12-31") -> pd.DataFrame:
    cache = Path("data/cache/fred_macro_full.csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print("  macro from cache")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    base = "https://api.stlouisfed.org/fred/series/observations"
    out = {}
    for sid, col in LONG_HISTORY_MACRO.items():
        for attempt in range(4):
            try:
                r = requests.get(base, params={"series_id": sid, "api_key": key, "file_type": "json",
                                               "observation_start": start, "observation_end": end}, timeout=30)
                j = r.json()
                if "observations" in j:
                    s = pd.Series({pd.to_datetime(o["date"]): (np.nan if o["value"] in (".", "") else float(o["value"]))
                                   for o in j["observations"]}).sort_index()
                    out[col] = s
                    print(f"  {sid:8} -> {col:8} {s.notna().sum():5d} obs")
                    break
                time.sleep(1.0)  # back off on rate limit
            except Exception as e:
                time.sleep(1.0)
                if attempt == 3:
                    print(f"  {sid} failed: {e}")
        time.sleep(0.7)  # throttle to avoid 429
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    df.to_csv(cache)
    return df


def main() -> dict:
    print("=" * 76)
    print("  RUN 2 — VIX-orthogonal MACRO (real credit/curve/funding) vs VIX, honest CPCV")
    print("=" * 76)
    _load_env()
    key = os.environ.get("FRED_API_KEY", "")
    assert key, "FRED_API_KEY missing (.env or env var)"

    from src.config import cfg
    from src.logging_setup import setup_logging
    setup_logging(level="ERROR", fmt="text")
    from src.data.market import download_all_market
    from src.features.engineering import engineer_features
    from src.models.fsi import FSIBuilder

    print("\n[1] Market data + features...")
    market = download_all_market()
    feat = engineer_features(market["sp500"], market["vix"])
    n = len(feat)
    tmask = np.zeros(n, dtype=bool)
    tmask[: int(n * 0.5)] = True
    feat, _ = FSIBuilder().build(feat, pd.DataFrame(), train_mask=tmask)

    print("[2] Fetching full-history FRED macro (real key)...")
    macro = fetch_macro(key)

    def ff(col):
        return macro[col].reindex(feat.index).ffill(limit=22) if col in macro else pd.Series(np.nan, index=feat.index)

    # VIX-orthogonal macro features (all causal: levels + backward diffs)
    feat["credit"] = ff("credit")
    feat["credit_chg_63"] = feat["credit"].diff(63)
    feat["slope2y"] = ff("slope2y")
    feat["slope_chg_63"] = feat["slope2y"].diff(63)
    feat["slope3m"] = ff("slope3m")
    feat["tbill3m"] = ff("tbill3m")
    feat["tbill_chg_63"] = feat["tbill3m"].diff(63)
    MACRO = [c for c in ["credit", "credit_chg_63", "slope2y", "slope_chg_63", "slope3m", "tbill_chg_63"]
             if feat[c].notna().mean() > 0.7]
    print(f"    usable VIX-orthogonal macro: {MACRO}")
    print(f"    credit coverage: {feat['credit'].notna().mean():.1%}  "
          f"({feat['credit'].dropna().index.min().date()} -> {feat['credit'].dropna().index.max().date()})")

    # Causal regime (HMM on train-head, filtered posteriors)
    print("[3] Causal HMM regime...")
    hf = ["log_ret", "vol_21d", "FSI"]
    Xh = feat[hf].fillna(0).to_numpy()
    th = int(len(Xh) * 0.5)
    sc = StandardScaler().fit(Xh[:th])
    Xs = sc.transform(Xh)
    best, bll = None, -np.inf
    for sd in range(10):
        try:
            mm = GaussianHMM(n_components=3, covariance_type="full", n_iter=120, random_state=sd)
            mm.fit(Xs[:th])
            ll = mm.score(Xs[:th])
            if ll > bll:
                bll, best = ll, mm
        except Exception:
            pass
    fil = filtered_state_proba(best, Xs)
    crisis = int(np.argsort(best.means_[:, 1])[-1])
    feat["c_prob_crisis"] = fil[:, crisis]

    # ── Honest sweep ────────────────────────────────────────────────
    print("\n[4] Honest CPCV sweep (VIX baseline vs price vs +macro vs +macro+regime)\n")
    PRICE = [c for c in ["vol_21d", "vix", "drawdown_63", "mom_21d", "mom_63d"] if c in feat.columns]

    def LR():
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=cfg.seed))

    cpcv = CPCVConfig(n_groups=6, n_test_groups=2, embargo=21)
    fwd_ret = feat["close"].pct_change().shift(-1)
    sweep = []
    winners = []

    for H in HORIZONS:
        for THv in THRESHOLDS:
            lab = crisis_label(feat["close"], H, THv)
            sets = {"price": PRICE, "price+macro": PRICE + MACRO,
                    "price+macro+regime": PRICE + MACRO + ["c_prob_crisis"]}
            allc = sorted(set(sum(sets.values(), [])))
            D = feat[allc].copy()
            D["label"] = lab.reindex(D.index)
            D = D.dropna(subset=allc)
            yv = D["label"]
            base = float(yv.dropna().mean())

            # VIX baseline via empirical CDF of train VIX (ranking only, no fit needed)
            vix_oos = run_cpcv(D[["vix"]], yv, _vix_factory, cpcv).oos_proba
            vix_pa = classification_metrics(yv.to_numpy(float), vix_oos.to_numpy(float))["pr_auc"]

            row = {"H": H, "thr": THv, "base": round(base, 4), "VIX": vix_pa}
            oos_by = {"VIX": vix_oos}
            for name, cols in sets.items():
                res = run_cpcv(D[cols], yv, make_calibrated_factory(LR), cpcv)
                pa = classification_metrics(yv.to_numpy(float), res.oos_proba.to_numpy(float))["pr_auc"]
                row[name] = pa
                oos_by[name] = res.oos_proba

            # gates for the best macro config
            best_set = max(sets, key=lambda s: (row[s] if row[s] == row[s] else -1))
            beat = (row[best_set] is not None and vix_pa is not None and row[best_set] > vix_pa)
            # PBO across {VIX, price, +macro, +macro+regime}
            from sklearn.metrics import average_precision_score
            names = list(oos_by)
            common = yv.dropna().index
            for nm in names:
                common = common.intersection(oos_by[nm].dropna().index)
            common = common.sort_values()
            S = 10
            sl = np.array_split(np.arange(len(common)), S)
            perf = np.full((S, len(names)), np.nan)
            yc = yv.reindex(common).to_numpy(float)
            for j, nm in enumerate(names):
                pv = oos_by[nm].reindex(common).to_numpy()
                for i, s in enumerate(sl):
                    ys, ps = yc[s], pv[s]
                    mk = ~(np.isnan(ys) | np.isnan(ps))
                    if mk.sum() >= 5 and len(np.unique(ys[mk])) > 1:
                        perf[i, j] = average_precision_score(ys[mk], ps[mk])
            PBO = probability_of_backtest_overfitting(perf, n_partitions=8).get("pbo")
            # Deflated Sharpe of the best macro config's de-risk strategy
            o = oos_by[best_set].reindex(feat.index)
            thr_q = float(o.dropna().quantile(0.85)) if o.notna().any() else 1.0
            sret = np.where(o.fillna(0).to_numpy() < thr_q, 1.0, 0.0) * fwd_ret.fillna(0).to_numpy()
            trials = []
            for nm in names:
                oo = oos_by[nm].reindex(feat.index)
                tq = float(oo.dropna().quantile(0.85)) if oo.notna().any() else 1.0
                rr = np.where(oo.fillna(0).to_numpy() < tq, 1.0, 0.0) * fwd_ret.fillna(0).to_numpy()
                sd_ = rr.std()
                trials.append(rr.mean() / sd_ if sd_ > 0 else 0.0)
            dsr = deflated_sharpe_ratio(sret, trials)["deflated_sharpe"]

            wins_all = bool(beat and PBO is not None and PBO < 0.5 and dsr is not None and dsr > 0.95)
            row.update({"best_set": best_set, "PBO": PBO, "DSR": dsr, "WINS_ALL_GATES": wins_all})
            sweep.append(row)
            if beat:
                winners.append(row)
            print(f"  H={H:>2} thr={THv:.0%} base={base:.3f} | VIX={vix_pa} | "
                  f"price={row['price']} +macro={row['price+macro']} +regime={row['price+macro+regime']} | "
                  f"PBO={PBO} DSR={dsr} {'★WINS' if wins_all else ''}")

    # ── Positive control: FSI vs STLFSI ──────────────────────────────
    fsi_r = None
    if "stlfsi" in macro:
        s = macro["stlfsi"].reindex(feat.index).ffill()
        idx = s.dropna().index
        if len(idx) > 200:
            fsi_r = round(float(np.corrcoef(feat["FSI"].reindex(idx).fillna(0), s.reindex(idx).fillna(0))[0, 1]), 4)

    any_win = any(r["WINS_ALL_GATES"] for r in sweep)
    print("\n" + "=" * 76)
    print("  VERDICT")
    print("=" * 76)
    print(f"  Config beats VIX on PR-AUC anywhere: {bool(winners)}")
    print(f"  Config clears ALL gates (>VIX & PBO<0.5 & DSR>0.95): {any_win}")
    print(f"  FSI vs STLFSI r = {fsi_r}")

    def _safe(o):
        import math
        return None if isinstance(o, float) and (math.isnan(o) or math.isinf(o)) else str(o)
    out = {"sweep": sweep, "any_beats_vix": bool(winners), "any_wins_all_gates": any_win,
           "fsi_vs_stlfsi_r": fsi_r, "macro_features": MACRO,
           "credit_source": "BAA10Y (Moody's Baa-10Y, full history)"}
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/run2_macro_edge.json", "w") as fh:
        json.dump(out, fh, indent=2, default=_safe)
    print("\n  Saved: outputs/run2_macro_edge.json")
    return out


class _VixECDF:
    def fit(self, X, y=None):
        self.s = np.sort(X.iloc[:, 0].to_numpy(float))
        return self

    def predict_proba(self, X):
        p = np.clip(np.searchsorted(self.s, X.iloc[:, 0].to_numpy(float)) / max(len(self.s), 1), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def _vix_factory():
    return _VixECDF()


if __name__ == "__main__":
    main()
