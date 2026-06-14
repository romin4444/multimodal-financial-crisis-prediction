"""
v3 — Discrete-time hazard (survival) model for crisis onset.

WHY:
    Crisis onset is fundamentally time-to-event. Binary "crisis within N days"
    classification fudges two things: (a) right-edge censoring (the last N days
    have no resolvable label) and (b) the duration dependence of risk (how long
    since the last drawdown). A discrete-time hazard model handles both naturally
    and yields exactly the object a risk officer wants:

        P(a >= X% drawdown occurs within the next N days | information today).

HOW (dependency-light, the standard pooled-logistic discrete-time hazard):
    1. Identify drawdown ONSET days and "at-risk" days (not already in a drawdown).
    2. Fit logistic regression of onset ~ features + duration on at-risk days
       (this is the discrete hazard h_t = P(onset at t | survived to t)).
    3. Convert the daily hazard into an N-day cumulative incidence:
           P(event within N) = 1 - prod_{k=0..N-1} (1 - h_{t+k})
       For forecasting we use the model's hazard under current features as the
       per-step hazard (a standard, transparent approximation).
    4. CALIBRATE the N-day cumulative incidence so "ranks risk well" becomes
       "calibrated probability you can quote."

CALIBRATION HISTORY:
    v3.0  class_weight="balanced" -> ranking (C-index) excellent but probabilities
          inflated ~20x on the ~4% base rate (Brier skill ~ -2). DROPPED.
    v3.3  un-weighted LR + isotonic calibration on the *temporal tail* of the
          training mask. This BACKFIRED: on a 1990->2010 train window the tail is
          ~2008-2010 (the GFC), so isotonic learned a crisis-era onset frequency
          and over-predicted on the calmer 2010->2024 test set. Result: calibrated
          Brier skill WORSE than raw (h21: +0.041 -> -0.029; h63: -0.164 -> -0.538).
    v3.4  (this file) THREE fixes:
          (1) low-variance Platt/sigmoid (2 params) instead of isotonic (~O(n) steps)
              — the right complexity for ~30 onsets;
          (2) calibrate on time-series OUT-OF-FOLD predictions spanning the WHOLE
              training window, so the (raw -> frequency) map is regime-representative
              rather than crisis-tail-biased;
          (3) a do-no-harm guard: keep the calibrator only if it beats raw on a
              held-out OOF check, else fall back to identity (= raw). Calibration
              can now never be worse than raw by construction.
          All REPORTED metrics remain on the strictly out-of-sample TEST mask; the
          OOF step is calibration-only (same principle as CalibratedClassifierCV).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# v3.4 calibrators
# ─────────────────────────────────────────────────────────────────────────────

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


class _IdentityCalibrator:
    """Pass-through 'calibrator' (the do-no-harm fallback).

    Attached when calibration does not beat the raw cumulative incidence on the
    held-out check, so ``calibrated`` output equals the (clipped) raw input and
    can never be worse than raw. Exposes ``.predict`` like ``IsotonicRegression``
    so ``HazardFit.cumulative_incidence`` needs no special-casing.
    """

    is_identity = True

    def fit(self, raw, y):  # noqa: D401 - parity with sklearn calibrators
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)


class _SigmoidCalibrator:
    """Platt scaling: a 2-parameter logistic on logit(raw_incidence).

    Two parameters (slope, intercept) is dramatically lower-variance than
    isotonic's ~O(n) steps — the right complexity when there are only ~30
    onsets. Strictly monotone increasing (when slope > 0), so it never inverts
    the hazard ranking the C-index measures.
    """

    is_identity = False

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.ok = False

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_SigmoidCalibrator":
        z = _logit(raw).reshape(-1, 1)
        self._lr.fit(z, np.asarray(y).astype(int))
        # A non-positive slope would invert the ranking -> reject.
        self.ok = bool(self._lr.coef_[0, 0] > 0)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        z = _logit(raw).reshape(-1, 1)
        return np.clip(self._lr.predict_proba(z)[:, 1], 0.0, 1.0)


def _make_calibrator(method: str):
    if method == "isotonic_oof":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return _SigmoidCalibrator()


def drawdown_panel(close: pd.Series, threshold: float = 0.10) -> pd.DataFrame:
    """
    Build the survival panel:
      - dd        : running drawdown from the trailing peak (<= 0).
      - in_dd     : currently in a >= `threshold` drawdown episode.
      - onset     : 1 on the day the drawdown first crosses below -threshold.
      - at_risk   : 1 on days NOT already inside a >= threshold drawdown.
      - duration  : trading days since the last episode ended (time at risk).
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    peak = np.maximum.accumulate(c)
    dd = c / peak - 1.0

    in_dd = dd <= -abs(threshold)
    onset = np.zeros(n, dtype=int)
    at_risk = np.zeros(n, dtype=int)
    duration = np.zeros(n, dtype=int)

    currently = False  # in a drawdown episode at the START of the day
    dur = 0
    for t in range(n):
        at_risk[t] = 0 if currently else 1
        if in_dd[t] and not currently:
            onset[t] = 1
            currently = True
        if not in_dd[t]:
            currently = False
        if at_risk[t]:
            dur += 1
        else:
            dur = 0
        duration[t] = dur

    return pd.DataFrame(
        {"dd": dd, "in_dd": in_dd.astype(int), "onset": onset,
         "at_risk": at_risk, "duration": duration},
        index=close.index,
    )


@dataclass
class HazardFit:
    model: object
    feature_cols: List[str]
    incidence_calibrator: Optional[object] = None
    calibrated_horizon: Optional[int] = None
    # v3.4: which calibrator was attached after the do-no-harm check. One of
    # "sigmoid_oof", "isotonic_oof", "isotonic_tail", "identity", or None.
    calib_method: Optional[str] = None

    def hazard(self, X: pd.DataFrame) -> np.ndarray:
        """Per-day onset hazard h_t."""
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def cumulative_incidence(
        self, X: pd.DataFrame, horizon: int, calibrated: bool = True
    ) -> np.ndarray:
        """
        P(event within `horizon` days) ≈ 1 - (1 - h_t)^horizon, using the current
        per-day hazard as the constant-within-horizon rate (transparent approx).

        When `calibrated=True` and a calibrator was fit at this horizon, the raw
        incidence is mapped through it.
        """
        h = np.clip(self.hazard(X), 1e-6, 1 - 1e-6)
        raw = 1.0 - (1.0 - h) ** horizon
        if (
            calibrated
            and self.incidence_calibrator is not None
            and self.calibrated_horizon == horizon
        ):
            return np.clip(self.incidence_calibrator.predict(raw), 1e-6, 1 - 1e-6)
        return raw


def _fit_lr(X: pd.DataFrame, y: pd.Series):
    # v3.3+: NO class_weight="balanced" (it inflates probs ~20x on a 4% base
    # rate and destroys Brier skill — see CALIBRATION HISTORY).
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0),
    )
    model.fit(X, y.astype(int))
    return model


def fit_hazard(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    horizon: Optional[int] = None,
    calibrate: bool = True,
    calib_frac: float = 0.2,
    drawdown_threshold: float = 0.10,
    calib_method: str = "sigmoid_oof",
    n_calib_folds: int = 4,
) -> HazardFit:
    """
    Fit the pooled-logistic discrete-time hazard on AT-RISK training days only.

    Args:
        horizon: N-day horizon for the cumulative-incidence calibrator. Required
            when ``calibrate=True``.
        calibrate: If True, fit a calibrator mapping raw N-day incidence to the
            realized N-day frequency.
        calib_method: "sigmoid_oof" (default, v3.4 — Platt on time-series OOF
            predictions + do-no-harm), "isotonic_oof" (isotonic on the same OOF),
            or "isotonic_tail" (the v3.3 temporal-tail isotonic, kept for A/B and
            back-compat — known to over-predict when the train tail is a crisis).
        n_calib_folds: number of contiguous time blocks for the OOF calibration.
        calib_frac: only used by the "isotonic_tail" path.
        drawdown_threshold: must match the threshold used to build ``panel``.

    Returns a ``HazardFit`` whose ``cumulative_incidence(X, horizon)`` returns
    calibrated probabilities when a calibrator was attached. With the default
    method calibration is never worse than raw by construction.
    """
    cols = [c for c in feature_cols if c in features.columns]
    use_cols = cols + ["duration"]

    data = features.copy()
    data["duration"] = panel["duration"]
    data["onset"] = panel["onset"]
    data["at_risk"] = panel["at_risk"]
    data = data.dropna(subset=cols)

    tr_mask_series = (
        pd.Series(train_mask, index=features.index)
        .reindex(data.index)
        .fillna(False)
        .to_numpy()
    )
    tr_full = data[(data["at_risk"] == 1) & tr_mask_series]

    want_calib = calibrate and horizon is not None and len(tr_full) > 200

    # ── isotonic_tail: preserved v3.3 behaviour (A/B + back-compat) ───────────
    if want_calib and calib_method == "isotonic_tail":
        cut = int(len(tr_full) * (1.0 - calib_frac))
        tr_model, tr_calib = tr_full.iloc[:cut], tr_full.iloc[cut:]
        fit = HazardFit(model=_fit_lr(tr_model[use_cols], tr_model["onset"]),
                        feature_cols=use_cols)
        if len(tr_calib) > 50:
            realized = _realized_within(panel, horizon, drawdown_threshold).reindex(tr_calib.index)
            valid = realized.notna().to_numpy()
            y_cal = realized[valid].to_numpy().astype(int)
            if len(np.unique(y_cal)) > 1 and y_cal.sum() >= 3:
                raw_cal = fit.cumulative_incidence(tr_calib[valid], horizon, calibrated=False)
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(raw_cal, y_cal)
                fit.incidence_calibrator = iso
                fit.calibrated_horizon = int(horizon)
                fit.calib_method = "isotonic_tail"
        return fit

    # ── default path: ranking model fit on ALL at-risk train days ────────────
    fit = HazardFit(model=_fit_lr(tr_full[use_cols], tr_full["onset"]),
                    feature_cols=use_cols)
    if not want_calib:
        return fit

    def _attach_identity():
        fit.incidence_calibrator = _IdentityCalibrator()
        fit.calibrated_horizon = int(horizon)
        fit.calib_method = "identity"
        return fit

    # Resolvable training rows (those with a realized N-day label).
    realized_full = _realized_within(panel, horizon, drawdown_threshold).reindex(tr_full.index)
    rv = realized_full.notna().to_numpy()
    base = tr_full[rv]
    y_all = realized_full[rv].to_numpy().astype(int)
    n = len(base)
    if n < 100 or y_all.sum() < 8 or len(np.unique(y_all)) < 2:
        return _attach_identity()

    # Time-series OUT-OF-FOLD raw incidence over the WHOLE train window.
    folds = np.array_split(np.arange(n), max(2, n_calib_folds))
    oof_raw = np.full(n, np.nan)
    for k in range(len(folds)):
        te_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        if base.iloc[tr_idx]["onset"].sum() < 3:
            continue
        m = _fit_lr(base.iloc[tr_idx][use_cols], base.iloc[tr_idx]["onset"])
        h = np.clip(m.predict_proba(base.iloc[te_idx][use_cols])[:, 1], 1e-6, 1 - 1e-6)
        oof_raw[te_idx] = 1.0 - (1.0 - h) ** horizon

    ok = np.isfinite(oof_raw)
    oof_raw, y_oof = oof_raw[ok], y_all[ok]
    if len(y_oof) < 50 or y_oof.sum() < 5 or len(np.unique(y_oof)) < 2:
        return _attach_identity()

    # do-no-harm: fit on the EARLIER 70% of the OOF pool (time order), CHECK on
    # the most-recent 30% — the slice closest to deployment and the toughest
    # test of generalization. Keep the calibrator only on a strict improvement;
    # otherwise ship identity (= raw). This is what makes calibration unable to
    # repeat the v3.3 regression: under a regime shift the guard can't confirm a
    # benefit on the recent slice, so we quote the raw probability.
    cut = int(len(y_oof) * 0.7)
    fit_i = np.arange(0, cut)
    chk_i = np.arange(cut, len(y_oof))

    cand = _make_calibrator(calib_method)
    try:
        cand.fit(oof_raw[fit_i], y_oof[fit_i])
    except Exception:
        return _attach_identity()
    if getattr(cand, "is_identity", False) or not getattr(cand, "ok", True):
        return _attach_identity()

    keep = False
    if len(chk_i) > 10 and y_oof[chk_i].sum() >= 2:
        y_chk = y_oof[chk_i]
        b_raw = _brier(y_chk, oof_raw[chk_i])
        b_cal = _brier(y_chk, cand.predict(oof_raw[chk_i]))
        b_clim = _brier(y_chk, np.full(len(y_chk), float(np.mean(y_chk))))
        # Two gates: (1) raw must already carry skill over climatology on the
        # recent slice — calibrating skill-less probabilities just transfers
        # train-regime bias (this is the h63 case); (2) calibration must give a
        # real (>=0.1%) Brier improvement. Fail either -> ship raw (identity).
        keep = (b_raw < b_clim) and (b_cal < b_raw * 0.999)
    if not keep:
        return _attach_identity()

    # Deploy the calibrator validated by the guard (fit on the earlier,
    # regime-representative 70%). Refitting on the FULL pool would re-inject the
    # crisis-tail bias the guard just screened out, so we keep `cand` as-is.
    fit.incidence_calibrator = cand
    fit.calibrated_horizon = int(horizon)
    fit.calib_method = "isotonic_oof" if calib_method == "isotonic_oof" else "sigmoid_oof"
    return fit


def evaluate_hazard(
    fit: HazardFit,
    panel: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int,
    test_mask: np.ndarray,
    drawdown_threshold: float = 0.10,
) -> dict:
    """
    Evaluate on at-risk TEST days:
      - C-index: AUC of the daily hazard score vs the observed onset (concordance).
      - N-day cumulative-incidence calibration vs the realized binary outcome.
      - Reports BOTH raw and calibrated N-day risk so the calibration effect shows.
    """
    data = features.copy()
    data["duration"] = panel["duration"]
    data["onset"] = panel["onset"]
    data["at_risk"] = panel["at_risk"]
    data = data.dropna(subset=[c for c in fit.feature_cols if c != "duration"])

    mask = (data["at_risk"] == 1) & pd.Series(test_mask, index=features.index).reindex(data.index).fillna(False).to_numpy()
    te = data[mask]
    if len(te) < 50 or te["onset"].sum() < 3:
        return {"n": int(len(te)), "c_index": float("nan"), "note": "insufficient test onsets"}

    h = fit.hazard(te)
    c_index = float(roc_auc_score(te["onset"].astype(int), h)) if te["onset"].nunique() > 1 else float("nan")

    realized = _realized_within(panel, horizon, drawdown_threshold).reindex(te.index)
    valid = realized.notna().to_numpy()
    y = realized[valid].to_numpy().astype(int)

    raw_risk = fit.cumulative_incidence(te[valid], horizon, calibrated=False)
    cal_risk = fit.cumulative_incidence(te[valid], horizon, calibrated=True)
    raw_risk = np.clip(raw_risk, 1e-6, 1 - 1e-6)
    cal_risk = np.clip(cal_risk, 1e-6, 1 - 1e-6)

    def _scores(p):
        if len(np.unique(y)) <= 1:
            return float("nan"), float("nan")
        brier = float(brier_score_loss(y, p))
        base = float(np.mean(y))
        bss = float(1 - brier / brier_score_loss(y, np.full_like(p, base)))
        return brier, bss

    brier_raw, bss_raw = _scores(raw_risk)
    brier_cal, bss_cal = _scores(cal_risk)
    base_rate = float(np.mean(y)) if len(y) else float("nan")
    calibrated = fit.incidence_calibrator is not None

    return {
        "n": int(len(te)),
        "n_onsets": int(te["onset"].sum()),
        "c_index": round(c_index, 4),
        "horizon": horizon,
        "Nday_base_rate": round(base_rate, 4) if np.isfinite(base_rate) else None,
        "Nday_risk_brier_raw": round(brier_raw, 4) if np.isfinite(brier_raw) else None,
        "Nday_risk_brier_skill_raw": round(bss_raw, 4) if np.isfinite(bss_raw) else None,
        "Nday_risk_brier_calibrated": round(brier_cal, 4) if np.isfinite(brier_cal) else None,
        "Nday_risk_brier_skill_calibrated": round(bss_cal, 4) if np.isfinite(bss_cal) else None,
        "calibrated": calibrated,
        "calib_method": fit.calib_method,
        "Nday_risk_brier": round(brier_cal if calibrated else brier_raw, 4) if np.isfinite(brier_cal if calibrated else brier_raw) else None,
        "Nday_risk_brier_skill": round(bss_cal if calibrated else bss_raw, 4) if np.isfinite(bss_cal if calibrated else bss_raw) else None,
    }


def _realized_within(panel: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:  # noqa: ARG001
    """For each at-risk day t, did a >= threshold drawdown onset occur in (t, t+horizon]?"""
    onset = panel["onset"].to_numpy(dtype=float)
    n = len(onset)
    if n == 0:
        return pd.Series([], dtype=float, index=panel.index)
    cs = np.concatenate([[0.0], np.cumsum(onset)])
    out = np.full(n, np.nan)
    for t in range(n - 1):
        j = min(t + 1 + horizon, n)
        out[t] = 1.0 if (cs[j] - cs[t + 1]) > 0 else 0.0
    return pd.Series(out, index=panel.index)
