#!/usr/bin/env python3
"""
Tiny, dependency-free converter: percent-script (.py with `# %%` cell markers)
-> Jupyter notebook (.ipynb v4). Used to build kaggle_frontier_benchmark.ipynb.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(py_path: Path, ipynb_path: Path) -> None:
    lines = py_path.read_text(encoding="utf-8").splitlines()
    cells = []
    cur_type, cur_lines = None, []

    def flush():
        if cur_type is None:
            return
        src = cur_lines[:]
        # strip trailing blank lines
        while src and src[-1].strip() == "":
            src.pop()
        if cur_type == "markdown":
            stripped = []
            for ln in src:
                stripped.append(ln[2:] if ln.startswith("# ") else (ln[1:] if ln.startswith("#") else ln))
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": _as_source(stripped)})
        else:
            if not src:
                return
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _as_source(src)})

    for ln in lines:
        if ln.startswith("# %% [markdown]"):
            flush(); cur_type, cur_lines = "markdown", []
        elif ln.startswith("# %%"):
            flush(); cur_type, cur_lines = "code", []
        else:
            cur_lines.append(ln)
    flush()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ipynb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {ipynb_path}  ({len(cells)} cells)")


def _as_source(lines):
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


if __name__ == "__main__":
    base = Path(__file__).parent
    stems = sys.argv[1:] or ["kaggle_frontier_benchmark", "kaggle_edge_and_multimodal"]
    for stem in stems:
        stem = stem.replace(".py", "").replace(".ipynb", "")
        src = base / f"{stem}.py"
        if src.exists():
            convert(src, base / f"{stem}.ipynb")
        else:
            print(f"skip (no .py): {src}")
