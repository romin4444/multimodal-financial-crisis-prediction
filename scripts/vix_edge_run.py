#!/usr/bin/env python3
"""
VIX-EDGE — can a VIX-ORTHOGONAL option-surface signal beat / improve on a VIX
threshold for forward equity-drawdown prediction, under the honest CPCV + PBO +
Deflated-Sharpe gates?

The headline signal is the Variance Risk Premium (VRP = IV^2 - RV^2), which needs
NO new data (Bollerslev-Tauchen-Zhou 2009). Term-structure (VIX9D/VIX3M) and SKEW
activate automatically if cached (data/cache/mkt_VIX9D.csv etc.); else skipped.

Two questions, both out-of-sample under CPCV:
  Q-REPLACE     : does any model beat the VIX-ECDF threshold on PR-AUC?
  Q-INCREMENTAL : does VIX + orthogonal residuals beat VIX-only-LR?  (the fair bar)

Built only on the pure, tested src/v3 modules + cached market data (offline-safe).
Author: Romin Patel.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.v3.cpcv import CPCVConfig, probability_of_backtest_overfitting, run_cpcv
from src.v3.deflated_sharpe import deflated_sharpe_ratio
from src.v3.metrics import classification_metrics
from src.v3.calibration import make_calibrated_factory
from src.v3.labeling import crisis_label
from src.v3.vix_orthogonal import build_vix_orthogonal, orthogonality_report

HORIZONS = [10, 21, 63]
THRESHOLDS = [0.07, 0.10]
CACHE = Path("data/cache")
SEED = 42


# ─── data / features (offline from cache, no statsmodels/hmm needed) ──────────

def _load(ticker: str) -> pd.DataFrame | None:
    p = CACHE / f"mkt_{ticker}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _max_drawdown(x: np.ndarray) -> float:
    peak = x.max()
    return 0.0 if peak == 0 else (x[-1] - peak) / peak


def build_features() -> tuple[pd.DataFrame, dict]:
    sp = _load("GSPC")
    vix = _load("VIX")
    if sp is None or vix is None:
        raise SystemExit("Cached S&P 500 / VIX not found in data/cache/")

    f = pd.DataFrame(index=sp.index)
    f["close"] = sp["Close"]
    f["vix"] = vix["Close"].reindex(f.index).ffill()
    f["log_ret"] = np.log(f["close"] / f["close"].shift(1))
    for w in (21, 63):
        f[f"vol_{w}d"] = f["log_ret"].rolling(w).std() * np.sqrt(252)
    f["drawdown_63"] = f["close"].rolling(63).apply(_max_drawdown, raw=True)
    for d in (21, 63):
        f[f"mom_{d}d"] = f["close"].pct_change(d)
    f = f.dropna(subset=["log_ret"])

    # optional option-surface feeds (offline -> usually absent, handled gracefully)
    def col(t):
        d = _load(t)
        return d["Close"].reindex(f.index).ffill() if d is not None else None
    feeds = {"vix9d": col("VIX9D"), "vix3m": col("VIX3M"), "skew": col("SKEW")}
    present = {k: (v is not None) for k, v in feeds.items()}

    ortho = build_vix_orthogonal(
        f, vix9d=feeds["vix9d"], vix3m=feeds["vix3m"], skew=feeds["skew"]
    )
    for c in ortho.columns:
        f[c] = ortho[c]
    return f, present


# ─── VIX-ECDF threshold baseline (the bar) ────────────────────────────────────

class _VixECDF:
    def fit(self, X, y=None):
        self.s = np.sort(X.iloc[:, 0].to_numpy(float))
        return self

    def predict_proba(self, X):
        p = np.clip(np.searchsorted(self.s, X.iloc[:, 0].to_numpy(float)) / max(len(self.s), 1),
                    1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def _vix_factory():
    return _VixECDF()


def LR():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    )


from src.json_utils import safe_json_default as _safe  # noqa: E402


def main() -> dict:
    print("=" * 80)
    print("  VIX-EDGE — VIX-orthogonal (Variance Risk Premium) vs VIX, honest CPCV")
    print("=" * 80)
    feat, present = build_features()
    print(f"\n[1] Features: {len(feat)} rows  "
          f"{feat.index.min().date()} -> {feat.index.max().date()}")
    print(f"    Option-surface feeds present: {present} "
          f"(VRP needs none; term-structure/SKEW skipped if absent)")

    # orthogonality of the headline signal
    orep = orthogonality_report(feat["vrp"], feat["vix"])
    print(f"[2] VRP vs VIX: corr={orep['corr_with_vix']}  "
          f"orthogonal_frac={orep['orthogonal_frac']} "
          f"(share of VRP variance VIX cannot explain)")

    PRICE = ["vol_21d", "vix", "drawdown_63", "mom_21d", "mom_63d"]
    VRP_RAW = [c for c in ["vrp", "vrp_ratio", "vol_of_vol"] if c in feat.columns]
    VRP_RES = [c for c in ["vrp_resid", "vrp_ratio_resid", "vol_of_vol_resid"] if c in feat.columns]
    TS = [c for c in feat.columns if c.startswith("ts_") and not c.endswith("_resid")]
    SK = [c for c in feat.columns if c.startswith("skew_") and not c.endswith("_resid")]

    cpcv = CPCVConfig(n_groups=6, n_test_groups=2, embargo=21)
    fwd_ret = feat["close"].pct_change().shift(-1)

    sweep = []
    print("\n[3] Honest CPCV sweep — PR-AUC (calibrated LR), per horizon/threshold\n")
    print(f"    {'H':>3} {'thr':>5} {'base':>6} | {'VIX*':>6} {'VIXlr':>6} {'price':>6} "
          f"{'+VRP':>6} {'+VRPres':>7} | {'beatBar':>7} {'beatVIX':>7} {'PBO':>5} {'DSR':>5}")

    for H in HORIZONS:
        for THv in THRESHOLDS:
            lab = crisis_label(feat["close"], H, THv)
            sets = {
                "VIX_lr": ["vix"],
                "price": PRICE,
                "price+VRP": PRICE + VRP_RAW + TS + SK,
                "price+VRP_resid": PRICE + VRP_RES,
            }
            allc = sorted(set(sum(sets.values(), [])))
            D = feat[allc].copy()
            D["label"] = lab.reindex(D.index)
            D = D.dropna(subset=allc)
            yv = D["label"]
            base = float(yv.dropna().mean())

            # VIX-ECDF threshold baseline (the bar to beat)
            vix_bar_oos = run_cpcv(D[["vix"]], yv, _vix_factory, cpcv).oos_proba
            vix_bar = classification_metrics(yv.to_numpy(float), vix_bar_oos.to_numpy(float))["pr_auc"]

            row = {"H": H, "thr": THv, "base": round(base, 4), "VIX_ECDF": vix_bar}
            oos_by = {"VIX_ECDF": vix_bar_oos}
            metr = {}
            for name, cols in sets.items():
                res = run_cpcv(D[cols], yv, make_calibrated_factory(LR), cpcv)
                m = classification_metrics(yv.to_numpy(float), res.oos_proba.to_numpy(float))
                row[name] = m["pr_auc"]
                metr[name] = m
                oos_by[name] = res.oos_proba

            # verdicts
            cand = row.get("price+VRP")
            cand_res = row.get("price+VRP_resid")
            best_cand = max([c for c in (cand, cand_res) if c is not None], default=None)
            beat_bar = bool(best_cand is not None and vix_bar is not None and best_cand > vix_bar)
            beat_vixlr = bool(best_cand is not None and row.get("VIX_lr") is not None
                              and best_cand > row["VIX_lr"])

            # PBO across the strategy set
            names = list(oos_by)
            common = yv.dropna().index
            for nm in names:
                common = common.intersection(oos_by[nm].dropna().index)
            common = common.sort_values()
            from sklearn.metrics import average_precision_score
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

            # Deflated Sharpe of the best candidate's de-risk strategy
            best_name = "price+VRP_resid" if (cand_res is not None and (cand is None or cand_res >= cand)) else "price+VRP"
            o = oos_by[best_name].reindex(feat.index)
            tq = float(o.dropna().quantile(0.85)) if o.notna().any() else 1.0
            sret = np.where(o.fillna(0).to_numpy() < tq, 1.0, 0.0) * fwd_ret.fillna(0).to_numpy()
            trials = []
            for nm in names:
                oo = oos_by[nm].reindex(feat.index)
                q = float(oo.dropna().quantile(0.85)) if oo.notna().any() else 1.0
                rr = np.where(oo.fillna(0).to_numpy() < q, 1.0, 0.0) * fwd_ret.fillna(0).to_numpy()
                sd_ = rr.std()
                trials.append(rr.mean() / sd_ if sd_ > 0 else 0.0)
            dsr = deflated_sharpe_ratio(sret, trials)["deflated_sharpe"]

            wins_all = bool(beat_bar and PBO is not None and PBO < 0.5 and dsr is not None and dsr > 0.95)
            row.update({
                "brier_skill_priceVRP": metr.get("price+VRP", {}).get("brier_skill"),
                "ece_priceVRP": metr.get("price+VRP", {}).get("ece"),
                "beat_bar": beat_bar, "beat_vix_lr": beat_vixlr,
                "PBO": PBO, "DSR": dsr, "WINS_ALL_GATES": wins_all,
            })
            sweep.append(row)
            print(f"    {H:>3} {THv:>5.0%} {base:>6.3f} | {str(vix_bar):>6} {str(row.get('VIX_lr')):>6} "
                  f"{str(row.get('price')):>6} {str(cand):>6} {str(cand_res):>7} | "
                  f"{str(beat_bar):>7} {str(beat_vixlr):>7} {str(PBO):>5} {str(dsr):>5} "
                  f"{'WINS' if wins_all else ''}")

    any_beat_bar = any(r["beat_bar"] for r in sweep)
    any_beat_vix = any(r["beat_vix_lr"] for r in sweep)
    any_win = any(r["WINS_ALL_GATES"] for r in sweep)

    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)
    print(f"  Q-REPLACE      any config beats VIX-ECDF on PR-AUC : {any_beat_bar}")
    print(f"  Q-INCREMENTAL  VIX+orthogonal beats VIX-only-LR    : {any_beat_vix}")
    print(f"  Clears ALL gates (>VIX & PBO<0.5 & DSR>0.95)       : {any_win}")

    out = {
        "experiment": "VIX-orthogonal (Variance Risk Premium) edge test",
        "data": {"rows": int(len(feat)),
                 "start": str(feat.index.min().date()), "end": str(feat.index.max().date()),
                 "option_surface_feeds_present": present},
        "vrp_orthogonality_vs_vix": orep,
        "horizons": HORIZONS, "thresholds": THRESHOLDS,
        "sweep": sweep,
        "any_beats_vix_ecdf": any_beat_bar,
        "any_beats_vix_lr_incremental": any_beat_vix,
        "any_wins_all_gates": any_win,
        "note": "Offline cached S&P500+VIX (1990-2024). VRP needs no feed; "
                "term-structure/SKEW activate when data/cache/mkt_VIX9D|VIX3M|SKEW.csv exist.",
    }
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/vix_edge_results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=_safe)
    print("\n  Saved: outputs/vix_edge_results.json")
    return out


if __name__ == "__main__":
    main()
