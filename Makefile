# FCPS — Financial Crisis Prediction System
# Common development commands

.PHONY: install install-dev train serve test test-cov lint format clean

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-asyncio pytest-cov httpx ruff black

# ── Pipeline ─────────────────────────────────────────────────────────────────
train:
	python scripts/train.py

train-news:
	python scripts/train.py --news-dir $(NEWS_DIR)

# ── API ───────────────────────────────────────────────────────────────────────
serve:
	python scripts/serve.py

serve-dev:
	python scripts/serve.py --reload

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
