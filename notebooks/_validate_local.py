#!/usr/bin/env python3
"""
Local validation of the Kaggle notebook's CORE logic (no GPU, no heavy installs).
Strips the pip-install calls, shrinks the date range for speed, execs the rest,
and asserts the honest benchmark table + PBO + DSR + hazard were produced.
Optional cells (TDA / Chronos / FinBERT) self-skip via their try/except guards.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

src = Path(__file__).parent / "kaggle_frontier_benchmark.py"
code = src.read_text(encoding="utf-8")

# Neutralise installs: make _pip a no-op (early return) without touching its signature
code = code.replace("def _pip(*pkgs):",
                    "def _pip(*pkgs):\n    return None  # neutralised for local validation")
# Shrink history for a fast smoke run
code = code.replace('START, END = "1990-01-01", "2024-12-31"',
                    'START, END = "2012-01-01", "2024-12-31"')
# Write outputs to the notebooks dir
code = code.replace('OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."',
                    'OUT = "."')

ns: dict = {"__name__": "__validate__"}
print("=== executing notebook core locally (this downloads data, ~1-2 min) ===")
exec(compile(code, str(src), "exec"), ns)

# ── Assertions ───────────────────────────────────────────────────────────────
assert "results" in ns and len(ns["results"]) >= 4, "benchmark results missing"
assert "tbl" in ns, "results table not built"
assert "PBO" in ns, "PBO not computed"
assert "DSR" in ns, "Deflated Sharpe not computed"
assert "c_index" in ns, "hazard C-index not computed"
vix = ns["results"].get("BASELINE VIX-threshold", {}).get("pr_auc")
assert vix is not None, "VIX baseline missing"
print("\n=== LOCAL VALIDATION PASSED ===")
print("models evaluated:", list(ns["results"].keys()))
print("VIX baseline PR-AUC:", vix, "| PBO:", ns["PBO"], "| hazard C-index:", round(float(ns["c_index"]), 4))
