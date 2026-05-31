# legacy/ — original capstone artifacts (archived)

These files are the **original** MBAI 5600G submission materials, kept for
provenance. They are **not** the canonical entry point — see the repository
root `README.md` for the current, modular pipeline (`src/`, `scripts/`).

| Path | What it is |
|---|---|
| `final.py` | The original 2,016-line single-file Kaggle notebook export. Superseded by the modular `src/` package. Kept so the refactor is auditable. |
| `notebooks/` | Five iterative Kaggle notebook versions (`notebook9a5760bcc1*.ipynb`). Development history. |
| `milestones/` | Course milestone deliverables (Milestone 1 & 2 PDFs, capstone explanation DOCX). |

**Canonical entry points (in the repo root):**
- `python scripts/demo_run.py` — 30-second end-to-end demo, no API key.
- `python scripts/v3_run.py` — honest walk-forward crisis backtest.
- `python scripts/direction_run.py` — stock-direction backtest.
- `python scripts/train.py` — full production pipeline.

Author: Romin Patel.
