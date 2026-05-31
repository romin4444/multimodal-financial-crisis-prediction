# legacy/ — original capstone artifacts (ARCHIVED — do not use)

> ⚠️ **Quarantined.** Everything in this folder is the **original** MBAI 5600G
> submission, kept only for provenance. It is **not** the reference
> implementation: it is **not maintained, not tested, and excluded from lint/CI**
> (`extend-exclude = ["legacy"]` in `pyproject.toml`). `final.py` here has the
> known evaluation-leakage issues documented in
> `docs/PROJECT_REVIEW_AND_ROADMAP.md`.
>
> 👉 The current, modular, leakage-free pipeline is in **`src/`** and **`scripts/`**.
> Start at the repository-root `README.md` or run `python scripts/demo_run.py`.

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
