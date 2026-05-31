"""
ARMA-GARCH volatility modelling — BIC model selection.
Huang & Luo (2024): GARCH, GJR-GARCH, EGARCH compared.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

from src.config import cfg
from src.logging_setup import get_logger

log = get_logger("models.garch")


class GARCHResult:
    """Container for a fitted GARCH model and its diagnostics."""

    def __init__(
        self,
        label: str,
        bic: float,
        aic: float,
        cond_vol: pd.Series,
        cond_var: pd.Series,
        lb_p: float,
        arch_p: float,
        converged: bool,
        result: object,
    ) -> None:
        self.label = label
        self.bic = bic
        self.aic = aic
        self.cond_vol = cond_vol
        self.cond_var = cond_var
        self.lb_p = lb_p
        self.arch_p = arch_p
        self.converged = converged
        self.result = result

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bic": round(self.bic, 2),
            "aic": round(self.aic, 2),
            "lb_p": round(self.lb_p, 4) if np.isfinite(self.lb_p) else None,
            "arch_p": round(self.arch_p, 4) if np.isfinite(self.arch_p) else None,
            "converged": self.converged,
        }


def select_garch(returns: pd.Series) -> Tuple[GARCHResult, List[GARCHResult]]:
    """
    Fit all configured GARCH specs and return the BIC-selected best model.

    Returns:
        (best_result, all_results): best is the BIC-minimising converged model.
    """
    log.info("Fitting GARCH-family specifications", extra={"n_specs": len(cfg.garch.specs)})
    results: List[GARCHResult] = [_fit_one(returns, spec.model_dump()) for spec in cfg.garch.specs]

    converged = [r for r in results if r.converged]
    if not converged:
        log.error("All GARCH specifications failed to converge — using rolling std fallback")
        return results[0], results

    best = min(converged, key=lambda r: r.bic)
    log.info("GARCH selected", extra=best.to_dict())
    return best, results


# ─── Private ─────────────────────────────────────────────────────────────────

def _fit_one(returns: pd.Series, spec: dict) -> GARCHResult:
    label = spec["label"]
    r100 = (returns * 100).dropna()

    try:
        am = arch_model(
            r100,
            mean="ARX",
            lags=1,
            vol=spec["vol"],
            p=spec["p"],
            o=spec["o"],
            q=spec["q"],
            dist="t",
            rescale=False,
        )
        res = am.fit(
            disp="off",
            options={"maxiter": cfg.garch.max_iter, "ftol": cfg.garch.ftol},
        )

        cond_vol = res.conditional_volatility / 100  # back to return scale
        cond_var = (cond_vol ** 2).rename("garch_var")

        std_resid = res.std_resid.dropna()
        lb_df = sm.stats.diagnostic.acorr_ljungbox(std_resid, lags=[10], return_df=True)
        lb_p = float(lb_df["lb_pvalue"].iloc[0])
        _, arch_p, *_ = het_arch(std_resid)

        log.info(
            "GARCH spec fitted",
            extra={
                "label": label,
                "bic": round(res.bic, 2),
                "aic": round(res.aic, 2),
                "lb_p": round(lb_p, 4),
                "arch_p": round(float(arch_p), 4),
            },
        )

        return GARCHResult(
            label=label,
            bic=float(res.bic),
            aic=float(res.aic),
            cond_vol=cond_vol,
            cond_var=cond_var,
            lb_p=lb_p,
            arch_p=float(arch_p),
            converged=True,
            result=res,
        )

    except Exception as exc:
        log.warning("GARCH spec failed", extra={"label": label, "error": str(exc)})
        # Rolling std fallback — clearly flagged as non-GARCH
        roll = returns.rolling(21).std().fillna(float(returns.std()))
        return GARCHResult(
            label=f"{label}[FALLBACK:rolling_std]",
            bic=float("inf"),
            aic=float("inf"),
            cond_vol=roll,
            cond_var=roll ** 2,
            lb_p=float("nan"),
            arch_p=float("nan"),
            converged=False,
            result=None,
        )
