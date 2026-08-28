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
from src.pipeline.ingest import DB_PATH
# pyrefly: ignore [missing-import]
from src.pipeline.classify import get_model_routing

ALL_MODELS = ["lgbm", "prophet", "sarima", "lstm", "dual"]
PREDICTION_TABLES = {
    "lgbm": "pred_lgbm",
    "prophet": "pred_prophet",
    "sarima": "pred_sarima",
    "lstm": "pred_lstm",
    "dual": "pred_dual",
}


def save_predictions(con, df, table_name: str) -> None:
    """Save predictions DataFrame to DuckDB."""
    if df is None or df.empty:
        logger.warning(f"  No predictions to save for {table_name}")
        return
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    logger.success(f"  Saved {table_name}: {len(df):,} rows")


def table_exists(con, table_name: str) -> bool:
    """Return whether a DuckDB table exists."""
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    return table_name in tables


def show_status() -> None:
    """Print model prediction-table status."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    routing = get_model_routing(con)
    logger.info("Model output status:")
    for model in ALL_MODELS:
        table = PREDICTION_TABLES[model]
        expected = len(routing[model if model != "dual" else "dual_phase"])
        if table_exists(con, table):
            rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            n_series = con.execute(f"SELECT COUNT(DISTINCT id) FROM {table}").fetchone()[0]
            state = "complete" if n_series >= expected else "partial"
            logger.info(
                f"  {model:8s} -> {table:12s}: {state:8s} "
                f"({n_series:,}/{expected:,} series, {rows:,} rows)"
            )
        else:
            logger.info(f"  {model:8s} -> {table:12s}: missing  (0/{expected:,} series)")
    if table_exists(con, "pred_ensemble"):
        rows = con.execute("SELECT COUNT(*) FROM pred_ensemble").fetchone()[0]
        n_series = con.execute("SELECT COUNT(DISTINCT id) FROM pred_ensemble").fetchone()[0]
        logger.info(f"  ensemble -> pred_ensemble: {rows:,} rows, {n_series:,} series")
    else:
        logger.info("  ensemble -> pred_ensemble: missing")
    con.close()


def should_skip_model(model_name: str, skip_existing: bool, expected_series: int) -> bool:
    """Return whether this model should be skipped because output already exists."""
    if not skip_existing:
        return False
    table = PREDICTION_TABLES[model_name]
    con = duckdb.connect(str(DB_PATH), read_only=True)
    exists = table_exists(con, table)
    n_series = (
        con.execute(f"SELECT COUNT(DISTINCT id) FROM {table}").fetchone()[0]
        if exists else 0
    )
    con.close()
    if exists and n_series >= expected_series:
        logger.info(f"[{model_name}] Skipping because {table} is complete.")
        return True
    if exists:
        logger.info(
            f"[{model_name}] {table} is partial ({n_series:,}/{expected_series:,} series). "
            "It will be rerun."
        )
    return False


def run_models(
    models_to_run: list[str],
    sample: int | None = None,
    skip_existing: bool = False,
):
    logger.info("=" * 60)
    logger.info("Supply Chain Forecasting — Model Training")
    logger.info("=" * 60)

    # Get routing using a read-only connection
    con = duckdb.connect(str(DB_PATH), read_only=True)
    routing = get_model_routing(con)
    con.close()

    t0 = time.time()

    # Apply sampling if requested
    if sample:
        logger.info(f"Sample mode: capping all model series at {sample}")
        routing = {m: ids[:sample] for m, ids in routing.items()}

    if skip_existing:
        models_to_run = [
            model for model in models_to_run
            if not should_skip_model(
                model,
                skip_existing=True,
                expected_series=len(routing[model if model != "dual" else "dual_phase"]),
            )
        ]
        if not models_to_run:
            logger.success("All requested model outputs already exist. Nothing to run.")
            show_status()
            return

    # LightGBM 
    if "lgbm" in models_to_run:
        logger.info("\n[LightGBM] Training…")
        # pyrefly: ignore [missing-import]
        from src.models.lgbm_model import train_lgbm, predict_lgbm
        con = duckdb.connect(str(DB_PATH), read_only=True)
        models = train_lgbm(con, series_ids=routing["lgbm"])
        preds = predict_lgbm(con, models, series_ids=routing["lgbm"])
        con.close()

        con = duckdb.connect(str(DB_PATH))
        save_predictions(con, preds, "pred_lgbm")
        con.close()

    # Prophet
    if "prophet" in models_to_run:
        logger.info("\n[Prophet] Training…")
        # pyrefly: ignore [missing-import]
        from src.models.prophet_model import train_predict_prophet
        con = duckdb.connect(str(DB_PATH), read_only=True)
        preds = train_predict_prophet(con, series_ids=routing["prophet"])
        con.close()

        con = duckdb.connect(str(DB_PATH))
        save_predictions(con, preds, "pred_prophet")
        con.close()

    # SARIMA
    if "sarima" in models_to_run:
        logger.info("\n[SARIMA] Training…")
        # pyrefly: ignore [missing-import]
        from src.models.sarima_model import train_predict_sarima
        con = duckdb.connect(str(DB_PATH), read_only=True)
        preds = train_predict_sarima(con, series_ids=routing["sarima"])
        con.close()

        con = duckdb.connect(str(DB_PATH))
        save_predictions(con, preds, "pred_sarima")
        con.close()

    # LSTM
    if "lstm" in models_to_run:
        logger.info("\n[LSTM] Training on MPS/GPU/CPU…")
        # pyrefly: ignore [missing-import]
        from src.models.lstm_model import train_lstm, predict_lstm
        con = duckdb.connect(str(DB_PATH), read_only=True)
        model, scaler, feats = train_lstm(con, series_ids=routing["lstm"])
        preds = predict_lstm(con, series_ids=routing["lstm"], model=model,
                             scaler=scaler, feature_names=feats)
        con.close()

        con = duckdb.connect(str(DB_PATH))
        save_predictions(con, preds, "pred_lstm")
        con.close()

    # Dual-Phase
    if "dual" in models_to_run:
        if routing["dual_phase"]:
            logger.info("\n[Dual-Phase] Training on intermittent/lumpy series…")
            # pyrefly: ignore [missing-import]
            from src.models.dual_phase import train_dual_phase, predict_dual_phase
            con = duckdb.connect(str(DB_PATH), read_only=True)
            clf, reg = train_dual_phase(con, series_ids=routing["dual_phase"])
            preds = predict_dual_phase(con, series_ids=routing["dual_phase"], clf=clf, reg=reg)
            con.close()

            con = duckdb.connect(str(DB_PATH))
            save_predictions(con, preds, "pred_dual")
            con.close()
        else:
            logger.warning("[Dual-Phase] No intermittent/lumpy series found. Skipping.")

    elapsed = time.time() - t0
    logger.success(f"\nAll models complete in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, help="Which models to train")
    parser.add_argument("--sample", type=int, default=None,
                        help="Cap each model at N series for quick testing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a model when its pred_* table already exists")
    parser.add_argument("--status", action="store_true",
                        help="Show which model prediction tables are already saved")
    args = parser.parse_args()
    if args.status:
        show_status()
    else:
        run_models(
            models_to_run=args.models,
            sample=args.sample,
            skip_existing=args.skip_existing,
        )
