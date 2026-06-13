#!/usr/bin/env python3
"""
v4-roadmap §2.3 — point-in-time (vintage / ALFRED) validation of the FSI.

WHY:
    The headline "FSI vs STLFSI r ≈ 0.82" was computed using LATEST-REVISED FRED
    data. A reviewer at an institution will (correctly) flag that: STLFSI4 and
    macro inputs are revised, so part of that correlation is hindsight.

    To convert the crown jewel result from "promising" to "reviewer-proof", we
    need r ≥ 0.70 on **vintage** data — i.e. validate the FSI against the
    STLFSI as it was known on each ``as_of`` date, using FRED inputs as they
    were known on each ``as_of`` date.

WHAT THIS SCRIPT DOES:
    For a rolling grid of as-of dates, pulls vintage FRED inputs via the
    ALFRED endpoint, rebuilds the FSI as of that date, and compares the
    rolling-window correlation against the contemporaneous STLFSI.

    Without ``FRED_API_KEY``, the script prints a clear "needs API key"
    message and exits 0 (CI-safe). With a key, it writes
    ``outputs/fsi_vintage_validation.json`` containing the as-of grid of r,
    NBER AUC, and the success-bar verdict.

USAGE:
    FRED_API_KEY=... python scripts/fsi_vintage_validate.py
    # or:
    # FRED_API_KEY=... python scripts/fsi_vintage_validate.py --as-of 2024-12-31

Author: Romin Patel.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--as-of",
        action="append",
        default=None,
        help="One or more as-of dates (YYYY-MM-DD). Repeat to add more. "
        "Defaults to year-ends 2010, 2015, 2020, 2024.",
    )
    p.add_argument(
        "--success-r",
        type=float,
        default=0.70,
        help="r threshold for the v4 roadmap §2.3 success bar (default 0.70).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv or sys.argv[1:])

    from src.config import cfg
    from src.data.fred import (
        align_fred_to_trading_days,
        download_fred_vintage,
        load_fred_key,
    )
    from src.data.market import download_all_market
    from src.features.engineering import engineer_features
    from src.json_utils import safe_json_default
    from src.logging_setup import setup_logging
    from src.models.fsi import FSIBuilder, validate_fsi

    setup_logging(level="WARNING", fmt="text")

    if not load_fred_key():
        print(
            "[skip] FRED_API_KEY not set — vintage validation needs ALFRED access.\n"
            "       Set the env var and re-run to populate "
            "outputs/fsi_vintage_validation.json.\n"
            "       Exiting 0 so CI stays green.",
        )
        return {"status": "skipped_no_api_key"}

    as_of_dates = args.as_of or ["2010-12-31", "2015-12-31", "2020-12-31", "2024-12-31"]
    print("=" * 76)
    print("  FCPS v4 — POINT-IN-TIME (VINTAGE) FSI vs STLFSI VALIDATION")
    print(f"  as_of grid: {as_of_dates}   |   success bar: r ≥ {args.success_r:.2f}")
    print("=" * 76)

    market = download_all_market()
    feat_base = engineer_features(market["sp500"], market["vix"])

    rows: list[dict] = []
    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)
        feat = feat_base.loc[:as_of_ts].copy()
        if len(feat) < 250:
            print(f"  [{as_of}] not enough history — skipping")
            continue

        vintage = download_fred_vintage(as_of=as_of_ts)
        fred_daily = align_fred_to_trading_days(vintage, feat.index)

        # FSI on vintage inputs (scaler fit on full history up to as_of — fine,
        # we're only validating against the contemporaneous STLFSI, not OOS-
        # forecasting).
        builder = FSIBuilder()
        n = len(feat)
        feat, _ = builder.build(
            feat, fred_daily, train_mask=np.ones(n, dtype=bool)
        )
        validity = validate_fsi(feat["FSI"], feat.join(fred_daily, how="left"))

        # If STLFSI was unavailable in this vintage (early years), skip.
        r = validity.get("stlfsi_r")
        if r is None:
            print(f"  [{as_of}] STLFSI not in this vintage — skipping")
            continue

        nber_auc = validity.get("nber_roc_auc")
        passes = bool(abs(r) >= args.success_r)
        rows.append(
            {
                "as_of": as_of,
                "stlfsi_r_vintage": round(float(r), 4),
                "nber_roc_auc_vintage": round(float(nber_auc), 4)
                if nber_auc is not None and np.isfinite(nber_auc)
                else None,
                "passes_roadmap_v4_2_3": passes,
                "n_days": int(len(feat)),
            }
        )
        print(
            f"  [{as_of}] r = {r:+.3f}  NBER-AUC = {nber_auc}  "
            f"{'PASS' if passes else 'FAIL'} v4 §2.3"
        )

    if not rows:
        print("  No vintage rows produced — every as_of either lacked data or STLFSI.")
        return {"status": "no_rows"}

    median_r = float(np.median([abs(r["stlfsi_r_vintage"]) for r in rows]))
    overall_passes = all(r["passes_roadmap_v4_2_3"] for r in rows)
    summary = {
        "as_of_grid": as_of_dates,
        "success_r_threshold": args.success_r,
        "rows": rows,
        "median_abs_r_vintage": round(median_r, 4),
        "overall_passes_v4_2_3": overall_passes,
        "note": (
            "Per v4 roadmap §2.3, the crown-jewel FSI vs STLFSI claim must "
            "hold on POINT-IN-TIME (vintage) data, not just on revised data. "
            "This artifact records that check."
        ),
    }

    out_path = cfg.paths.output_dir / "fsi_vintage_validation.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=safe_json_default)
    print(f"\n  median |r| vintage = {median_r:.3f}   overall pass: {overall_passes}")
    print(f"  Saved: {out_path}")
    return summary


if __name__ == "__main__":
    main()
