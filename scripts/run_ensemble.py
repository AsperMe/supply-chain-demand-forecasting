#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger
import duckdb

# pyrefly: ignore [missing-import]
from src.pipeline.ingest import DB_PATH
# pyrefly: ignore [missing-import]
from src.ensemble.stacker import train_ensemble, predict_ensemble
# pyrefly: ignore [missing-import]
from src.ensemble.evaluate import evaluate_all_models


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    return table in tables


def load_saved_ensemble(con: duckdb.DuckDBPyConnection):
    """Load saved ensemble predictions for report generation."""
    if not _table_exists(con, "pred_ensemble"):
        raise RuntimeError("pred_ensemble does not exist yet. Run the ensemble first.")
    return con.execute("SELECT * FROM pred_ensemble").df()


def run_ensemble(evaluate_only: bool = False):
    logger.info("=" * 60)
    logger.info("Supply Chain Forecasting — Ensemble + Evaluation")
    logger.info("=" * 60)

    if evaluate_only:
        logger.info("\n[1/1] Loading saved pred_ensemble for evaluation…")
        con = duckdb.connect(str(DB_PATH), read_only=True)
        preds = load_saved_ensemble(con)
        con.close()
    else:
        # Step 1: Train meta-learner
        logger.info("\n[1/3] Training ensemble meta-learner…")
        con = duckdb.connect(str(DB_PATH), read_only=True)
        meta = train_ensemble(con)
        con.close()

        # Step 2: Generate ensemble predictions 
        logger.info("\n[2/3] Generating ensemble predictions…")
        con = duckdb.connect(str(DB_PATH), read_only=True)
        preds = predict_ensemble(con, meta)
        con.close()

        # Step 3: Save ensemble predictions
        if not preds.empty:
            con = duckdb.connect(str(DB_PATH))
            con.execute("DROP TABLE IF EXISTS pred_ensemble")
            con.execute("CREATE TABLE pred_ensemble AS SELECT * FROM preds")
            logger.success(f"  Saved pred_ensemble: {len(preds):,} rows")
            con.close()

    # Step 4: Evaluate all models
    logger.info("\nEvaluating all models…")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    metrics = evaluate_all_models(preds, con)
    con.close()

    logger.success("\nEnsemble complete. Results saved to reports/model_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Reuse saved pred_ensemble and generate reports only",
    )
    args = parser.parse_args()
    run_ensemble(evaluate_only=args.evaluate_only)
