#!/usr/bin/env python3
"""
Training script — runs the full pipeline and saves all artifacts.
Usage:
    python scripts/train.py
    python scripts/train.py --news-dir /path/to/news/csvs
    python scripts/train.py --config /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FCPS training pipeline")
    parser.add_argument("--news-dir", type=Path, default=None,
                        help="Directory containing news CSV files")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml (defaults to project root)")
    args = parser.parse_args()

    if args.config:
        import os
        os.environ["CONFIG_PATH"] = str(args.config)

    from src.pipeline import run_pipeline

    result = run_pipeline(news_dir=args.news_dir)
    print(f"\nPipeline completed in {result.runtime_seconds / 60:.1f} minutes")
    print(f"Integration CSV rows: {len(result.feat):,}")


if __name__ == "__main__":
    main()
