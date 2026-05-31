"""
SHAP explainability — TreeExplainer for Random Forest fusion model.
Identifies whether price or sentiment drives each crisis (Lundberg 2020).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.logging_setup import get_logger
from src.models.fusion import TrainedFusion

log = get_logger("analysis.shap")


def run_shap(
    fusion: pd.DataFrame,
    trained: TrainedFusion,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Compute SHAP attributions using Random Forest TreeExplainer.

    Returns:
        shap_results: dict with 'by_crisis' mapping crisis name → feature importance Series.
        X: Feature DataFrame used for SHAP computation.
    """
    log.info("Computing SHAP attributions")
    try:
        import shap  # lazy — optional extra: pip install -e ".[explain]"
    except ImportError:
        log.warning("shap not installed — SHAP attribution skipped (pip install -e '.[explain]')")
        fcols = trained.feature_cols
        return {"by_crisis": {}}, pd.DataFrame(fusion[fcols].values, columns=fcols, index=fusion.index)

    fcols = trained.feature_cols
    X = pd.DataFrame(fusion[fcols].values, columns=fcols, index=fusion.index)

    rf = trained.models.get("Random Forest")
    if rf is None:
        log.warning("Random Forest not in trained models — SHAP skipped")
        return {"by_crisis": {}}, X

    shap_results: dict = {"by_crisis": {}}

    try:
        explainer = shap.TreeExplainer(rf)
        global_values = explainer.shap_values(X)
        if isinstance(global_values, list):
            global_values = global_values[1]  # crisis class (index 1)
        elif global_values.ndim == 3:
            global_values = global_values[:, :, 1]
        shap_results["global_values"] = global_values
        shap_results["feature_cols"] = fcols
    except Exception as exc:
        log.warning("Global SHAP computation failed", extra={"error": str(exc)})
        explainer = None

    if explainer is None:
        return shap_results, X

    from src.config import cfg
    for crisis_name, (cs, ce) in cfg.crisis_window_tuples.items():
        mask = (X.index >= cs) & (X.index <= ce)
        if mask.sum() == 0:
            continue
        try:
            v = explainer.shap_values(X[mask])
            if isinstance(v, list):
                v = v[1]
            elif v.ndim == 3:
                v = v[:, :, 1]
            importance = pd.Series(
                np.abs(v).mean(axis=0), index=fcols
            ).sort_values(ascending=False)
            shap_results["by_crisis"][crisis_name] = importance
            top3 = importance.head(3).to_dict()
            log.info(
                "SHAP by crisis",
                extra={"crisis": crisis_name, "top3": {k: round(v, 4) for k, v in top3.items()}},
            )
        except Exception as exc:
            log.warning("SHAP failed for crisis window", extra={"crisis": crisis_name, "error": str(exc)})

    return shap_results, X
