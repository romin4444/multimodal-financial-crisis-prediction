# FCPS — Financial Crisis Prediction System
# Author: Romin Patel
# Common development commands

.PHONY: help install install-all install-lock demo train serve test test-cov lint format clean

help:
	@echo "FCPS make targets:"
	@echo "  make install      Editable install, core deps only (no torch)"
	@echo "  make install-all  Editable install incl. NLP (torch) + explain + dev"
	@echo "  make install-lock Reproducible install from requirements.lock"
	@echo "  make demo         >>> 30-second end-to-end run on cached/synthetic data, NO API key"
	@echo "  make train        Full pipeline (needs FRED_API_KEY + optional news dir)"
	@echo "  make serve        Launch the prediction API on :8000"
	@echo "  make test         Run the test suite"
	@echo "  make test-cov     Run tests with coverage report"

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -e .

install-all:
	pip install -e ".[all]"

install-lock:
	pip install -r requirements.lock && pip install -e . --no-deps

# ── 30-second demo (no external APIs, no FRED key needed) ──────────────────────
demo:
	python scripts/demo_run.py

# ── Honest v3 backtests ─────────────────────────────────────────────────────────
v3:
	python scripts/v3_run.py

direction:
	python scripts/direction_run.py

# ── Full pipeline / API ─────────────────────────────────────────────────────────
train:
	python scripts/train.py

serve:
	python scripts/serve.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest -q --tb=short

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage **/__pycache__ *.egg-info
