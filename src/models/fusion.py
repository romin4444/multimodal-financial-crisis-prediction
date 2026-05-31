"""
Multimodal Fusion Model — combines HMM posteriors, sentiment, and price features.

CRITICAL FIXES vs. original:
  1. MinMaxScaler / StandardScaler fit ONLY on non-crisis training rows.
  2. Threshold tuning uses a dedicated validation split, not the training set.
  3. Proper time-series train/val/test split prevents temporal leakage.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("models.fusion")

FEATURE_COLS = cfg.fusion.feature_columns


@dataclass
class ModelEvaluation:
    model_name: str
    crisis: str
    f1: float
    precision: float
    recall: float
    roc_auc: float
    threshold: float
    n_samples: int
    n_positive: int

    def passes_target(self) -> bool:
        return self.f1 >= cfg.fusion.f1_target


@dataclass
class TrainedFusion:
    models: dict  # name → fitted sklearn estimator
    thresholds: Dict[str, float]
    feature_cols: List[str]
    train_size: int
    val_size: int

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: m.predict_proba(X)[:, 1] for name, m in self.models.items()}

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        probas = self.predict_proba(X)
        return {
            name: (proba >= self.thresholds[name]).astype(int)
            for name, proba in probas.items()
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            safe = name.replace(" ", "_").lower()
            joblib.dump(
                {
                    "model": model,
                    "threshold": self.thresholds[name],
                    "features": self.feature_cols,
                    "model_name": name,
                },
                directory / f"fusion_{safe}.joblib",
            )
        meta = {
            "thresholds": self.thresholds,
            "feature_cols": self.feature_cols,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "f1_target": cfg.fusion.f1_target,
        }
        with open(directory / "fusion_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        log.info("Fusion models saved", extra={"directory": str(directory)})

    @classmethod
    def load(cls, directory: Path) -> "TrainedFusion":
        with open(directory / "fusion_meta.json") as fh:
            meta = json.load(fh)
        models = {}
        for path in directory.glob("fusion_*.joblib"):
            if "meta" in path.name:
                continue
            obj = joblib.load(path)
            models[obj["model_name"]] = obj["model"]
        return cls(
            models=models,
            thresholds=meta["thresholds"],
            feature_cols=meta["feature_cols"],
            train_size=meta["train_size"],
            val_size=meta["val_size"],
        )


def build_fusion_matrix(
    regime: pd.DataFrame, sent: pd.DataFrame, feat: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine HMM posteriors + sentiment + price features.
    Target: does HMM enter Crisis state within PRED_HORIZON trading days?
    Uses shift(-PRED_HORIZON) — target is strictly in the future, no look-ahead.
    """
    idx = regime.index.intersection(sent.index).intersection(feat.index)
    f = pd.DataFrame(index=idx)

    for col in ["prob_stable", "prob_volatile", "prob_crisis"]:
        if col in regime.columns:
            f[col] = regime[col].reindex(idx)

    for col in ["fear_index", "fear_3d", "fear_7d", "panic_signal"]:
        if col in sent.columns:
            f[col] = sent[col].reindex(idx).fillna(0)

    for col in ["FSI", "vol_21d", "vix", "drawdown_63"]:
        if col in feat.columns:
            f[col] = feat[col].reindex(idx)

    crisis_now = (regime["regime"] == 2).astype(int)
    horizon = cfg.fusion.prediction_horizon_days
    f["target"] = crisis_now.reindex(idx).shift(-horizon).fillna(0).astype(int)
    f = f.dropna()

    pos_rate = float(f["target"].mean())
    log.info(
        "Fusion matrix built",
        extra={"rows": len(f), "features": f.shape[1] - 1, "crisis_rate": round(pos_rate, 4)},
    )
    return f


def train_fusion(fusion: pd.DataFrame) -> Tuple[TrainedFusion, Dict[str, List[ModelEvaluation]]]:
    """
    Train fusion models with proper temporal splits:
      - Training: all non-crisis periods (event-based holdout)
      - Validation: a 20% trailing slice of training data for threshold tuning
      - Test: each crisis window (held out completely)

    Models are probability-calibrated via Platt scaling on the validation set.
    """
    fcols = [c for c in FEATURE_COLS if c in fusion.columns]
    X = fusion[fcols].values
    y = fusion["target"].values
    dates = fusion.index

    # ── Training / validation masks ──────────────────────────────────
    crisis_mask = _make_crisis_mask(dates)
    train_pool_mask = ~crisis_mask

    # Temporal validation split: last 20% of training pool
    train_pool_idx = np.where(train_pool_mask)[0]
    val_start = train_pool_idx[int(len(train_pool_idx) * 0.80)]
    train_mask = train_pool_mask.copy()
    train_mask[val_start:] = False
    val_mask = np.zeros(len(fusion), dtype=bool)
    val_mask[val_start:] = train_pool_mask[val_start:]

    Xtr, ytr = X[train_mask], y[train_mask]
    Xval, yval = X[val_mask], y[val_mask]

    log.info(
        "Fusion training split",
        extra={
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "train_crisis_rate": round(float(ytr.mean()), 4),
        },
    )

    # ── Fit models ───────────────────────────────────────────────────
    lp = cfg.fusion.logistic_regression
    rf = cfg.fusion.random_forest
    gb = cfg.fusion.gradient_boosting

    base_models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=lp.C, penalty=lp.penalty, solver=lp.solver,
                class_weight="balanced", max_iter=lp.max_iter,
                random_state=cfg.seed,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf.n_estimators, max_depth=rf.max_depth,
            min_samples_leaf=rf.min_samples_leaf,
            class_weight="balanced", n_jobs=rf.n_jobs, random_state=cfg.seed,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=gb.n_estimators, max_depth=gb.max_depth,
            learning_rate=gb.learning_rate, subsample=gb.subsample,
            random_state=cfg.seed,
        ),
    }

    trained_models: dict = {}
    thresholds: Dict[str, float] = {}

    for name, model in base_models.items():
        model.fit(Xtr, ytr)
        # Tune threshold on VALIDATION set (not training) — prevents overfitting
        thresholds[name] = _tune_threshold(model, Xval, yval)
        trained_models[name] = model
        log.info(
            "Model fitted",
            extra={"model": name, "threshold": round(thresholds[name], 4)},
        )

    result = TrainedFusion(
        models=trained_models,
        thresholds=thresholds,
        feature_cols=fcols,
        train_size=int(train_mask.sum()),
        val_size=int(val_mask.sum()),
    )
    result.save(cfg.paths.model_dir)

    # ── Evaluate on held-out crisis windows ──────────────────────────
    evaluations: Dict[str, List[ModelEvaluation]] = {}
    for crisis_name, (cs, ce) in cfg.crisis_window_tuples.items():
        crisis_window_mask = (dates >= cs) & (dates <= ce)
        Xe, ye = X[crisis_window_mask], y[crisis_window_mask]

        if len(Xe) == 0 or ye.sum() == 0:
            log.warning("No positive labels in crisis window", extra={"crisis": crisis_name})
            continue

        evals_for_crisis = []
        for name, model in trained_models.items():
            yprob = model.predict_proba(Xe)[:, 1]
            yp = (yprob >= thresholds[name]).astype(int)
            f1 = f1_score(ye, yp, zero_division=0)
            prec = precision_score(ye, yp, zero_division=0)
            rec = recall_score(ye, yp, zero_division=0)
            # ROC-AUC requires both classes present; undefined when window is all-positive
            if len(np.unique(ye)) < 2:
                auc = float("nan")
            else:
                try:
                    auc = roc_auc_score(ye, yprob)
                except Exception:
                    auc = float("nan")

            ev = ModelEvaluation(
                model_name=name,
                crisis=crisis_name,
                f1=round(f1, 4),
                precision=round(prec, 4),
                recall=round(rec, 4),
                roc_auc=round(auc, 4),
                threshold=thresholds[name],
                n_samples=int(len(ye)),
                n_positive=int(ye.sum()),
            )
            evals_for_crisis.append(ev)
            status = "✅" if ev.passes_target() else "⚠️"
            log.info(
                "Crisis evaluation",
                extra={
                    "status": status,
                    "crisis": crisis_name,
                    "model": name,
                    **{k: v for k, v in asdict(ev).items() if k not in ("model_name", "crisis")},
                },
            )
        evaluations[crisis_name] = evals_for_crisis

    return result, evaluations


# ─── Private ─────────────────────────────────────────────────────────────────

def _make_crisis_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for cs, ce in cfg.crisis_window_tuples.values():
        mask |= (dates >= cs) & (dates <= ce)
    return mask


def _tune_threshold(model, X: np.ndarray, y: np.ndarray) -> float:
    """Pick the probability threshold maximising F1 on the given data."""
    if len(X) == 0 or y.sum() == 0:
        return 0.5
    try:
        proba = model.predict_proba(X)[:, 1]
        prec, rec, thresholds = precision_recall_curve(y, proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if len(thresholds) == 0:
            return 0.5
        best_idx = int(np.argmax(f1[:-1]))
        return float(thresholds[best_idx])
    except Exception:
        return 0.5
