"""
v3 — Topological Data Analysis (TDA) features for crisis early-warning.

WHY (the 2025 frontier):
    Topological ML for Financial Crisis Detection (MDPI Computers, 2025) reports
    early-warning signals from PERSISTENT HOMOLOGY of sliding return windows,
    with ~34-day average lead time on US equities. This module reproduces that
    family of features so we can benchmark it inside our honest CPCV harness:

      - Take a sliding window of (multivariate or Takens-embedded) returns.
      - Compute the persistence diagram (Vietoris-Rips).
      - Summarise it into scalar early-warning features: total persistence,
        max persistence (H1), and persistence-landscape L2 norm.

Dependencies are OPTIONAL and imported lazily:
    pip install giotto-tda      (preferred)   OR   pip install ripser
If neither is installed, build_tda_features() returns an empty frame and logs a
clear message — the rest of the pipeline runs unaffected.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.logging_setup import get_logger

log = get_logger("v3.tda")

TDA_FEATURE_COLS = ["tda_total_persistence", "tda_max_persistence_h1", "tda_landscape_l2"]


def _takens_embedding(x: np.ndarray, dim: int = 3, delay: int = 1) -> np.ndarray:
    """Delay-coordinate (Takens) embedding of a 1-D series into R^dim."""
    n = len(x) - (dim - 1) * delay
    if n <= 0:
        return np.empty((0, dim))
    return np.column_stack([x[i * delay : i * delay + n] for i in range(dim)])


def _diagram_summaries_giotto(window_points: np.ndarray) -> Optional[dict]:
    try:
        from gtda.homology import VietorisRipsPersistence
    except Exception:
        return None
    vr = VietorisRipsPersistence(homology_dimensions=(0, 1), n_jobs=1)
    diags = vr.fit_transform(window_points[None, :, :])[0]  # (n_features, 3): birth, death, dim
    return _summarise(diags)


def _diagram_summaries_ripser(window_points: np.ndarray) -> Optional[dict]:
    try:
        from ripser import ripser
    except Exception:
        return None
    res = ripser(window_points, maxdim=1)["dgms"]
    rows = []
    for dim, dgm in enumerate(res):
        for b, d in dgm:
            if np.isfinite(d):
                rows.append((b, d, dim))
    return _summarise(np.array(rows) if rows else np.empty((0, 3)))


def _summarise(diags: np.ndarray) -> dict:
    """Scalar early-warning summaries of a persistence diagram."""
    if diags.size == 0:
        return {c: 0.0 for c in TDA_FEATURE_COLS}
    births, deaths, dims = diags[:, 0], diags[:, 1], diags[:, 2]
    lifetimes = np.clip(deaths - births, 0, None)
    total_persistence = float(np.nansum(lifetimes))
    h1 = lifetimes[dims == 1]
    max_h1 = float(np.nanmax(h1)) if h1.size else 0.0
    # crude persistence-landscape L2: sqrt(sum of squared lifetimes)
    landscape_l2 = float(np.sqrt(np.nansum(lifetimes ** 2)))
    return {
        "tda_total_persistence": total_persistence,
        "tda_max_persistence_h1": max_h1,
        "tda_landscape_l2": landscape_l2,
    }


def tda_available() -> bool:
    try:
        import gtda  # noqa: F401
        return True
    except Exception:
        try:
            import ripser  # noqa: F401
            return True
        except Exception:
            return False


def build_tda_features(
    returns: pd.Series,
    window: int = 63,
    embed_dim: int = 3,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Sliding-window TDA early-warning features, indexed at the END of each window
    (so they are causal: window [t-window+1, t] -> feature at t).

    Returns a DataFrame with TDA_FEATURE_COLS, or an empty frame if no TDA
    library is installed.
    """
    if not tda_available():
        log.warning("No TDA backend (pip install giotto-tda OR ripser) — TDA features skipped")
        return pd.DataFrame(columns=TDA_FEATURE_COLS, index=returns.index)

    use_giotto = True
    try:
        import gtda  # noqa: F401
    except Exception:
        use_giotto = False

    r = returns.to_numpy(dtype=float)
    idx = returns.index
    n = len(r)
    rows: List[dict] = []
    row_index = []
    for t in range(window, n, stride):
        seg = r[t - window : t]
        seg = np.nan_to_num(seg, nan=0.0)
        pts = _takens_embedding(seg, dim=embed_dim, delay=1)
        if pts.shape[0] < embed_dim + 1:
            continue
        summ = _diagram_summaries_giotto(pts) if use_giotto else _diagram_summaries_ripser(pts)
        if summ is None:
            continue
        rows.append(summ)
        row_index.append(idx[t])

    if not rows:
        return pd.DataFrame(columns=TDA_FEATURE_COLS, index=returns.index)

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(row_index))
    # reindex to the full series, forward-filling between strides (causal)
    df = df.reindex(idx).ffill()
    log.info("TDA features built", extra={"backend": "giotto" if use_giotto else "ripser",
                                          "windows": len(rows)})
    return df[TDA_FEATURE_COLS]
