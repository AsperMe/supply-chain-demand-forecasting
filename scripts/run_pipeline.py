#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
import duckdb

# pyrefly: ignore [missing-import]
from src.pipeline.ingest import ingest, DB_PATH
# pyrefly: ignore [missing-import]
from src.pipeline.features import build_features
# pyrefly: ignore [missing-import]
from src.pipeline.classify import classify_demand, get_model_routing


def run_pipeline(force: bool = False, sample: int | None = None):
    logger.info("=" * 60)
    logger.info("Supply Chain Forecasting — Data Pipeline")
    logger.info("=" * 60)

    t0 = time.time()

    # Step 1: Ingest
    logger.info("\n[1/3] Data Ingestion")
    con = ingest(force=force)

    # Step 2: Feature Engineering
    logger.info("\n[2/3] Feature Engineering")
    if sample:
        sample_ids = con.execute(
            f"SELECT DISTINCT id FROM master ORDER BY RANDOM() LIMIT {sample}"
        ).df()["id"].tolist()
        logger.info(f"  Sample mode: {len(sample_ids)} series")
    else:
        sample_ids = None

    build_features(con, sample_ids=sample_ids)

    # Step 3: Demand Classification
    logger.info("\n[3/3] Demand Classification")
    classify_demand(con)

    routing = get_model_routing(con)
    logger.info("\nModel routing summary:")
    for model, ids in routing.items():
        logger.info(f"  {model:15s}: {len(ids):,} series")

    elapsed = time.time() - t0
    logger.success(f"\nPipeline complete in {elapsed:.1f}s")
    logger.info(f"Database: {DB_PATH}")
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only N series (for fast testing)")
    args = parser.parse_args()
    run_pipeline(force=args.force, sample=args.sample)
