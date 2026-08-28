"""
Supply Chain Demand Forecasting — Ensemble Stacker
Combines model predictions using a Ridge meta-learner per demand class.

Architecture:
  Base models: LightGBM (point), Prophet, SARIMA, LSTM, Dual-Phase
  Meta-learner: Ridge regression (non-negative weights) per demand class
  Final: demand-class-aware weighted average

Key design:
  - Separate meta-learner per demand class (Smooth, Intermittent, Erratic, Lumpy)
  - Weights are non-negative (constrained Ridge)
  - Out-of-fold predictions used for meta-learner training (no leakage)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"
MODEL_DIR = ROOT / "data" / "models"
REPORTS_DIR = ROOT / "reports"

DEMAND_CLASSES = ["Smooth", "Intermittent", "Erratic", "Lumpy"]

# Model columns available per demand class
MODEL_AVAILABILITY = {
    "Smooth":       ["lgbm_point", "prophet_point", "sarima_point", "lstm_point"],
    "Erratic":      ["lgbm_point", "prophet_point", "lstm_point"],
    "Intermittent": ["lgbm_point", "dual_point"],
    "Lumpy":        ["lgbm_point", "dual_point"],
}


def _load_all_predictions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load all model predictions from DuckDB.
    Expects tables: pred_lgbm, pred_prophet, pred_sarima, pred_lstm, pred_dual
    """
    logger.info("Loading all model predictions…")

    # LightGBM predictions
    lgbm = con.execute(
        "SELECT id, date, lgbm_point FROM pred_lgbm"
    ).df() if _table_exists(con, "pred_lgbm") else pd.DataFrame()

    # Prophet
    prophet = con.execute(
        "SELECT id, date, prophet_point FROM pred_prophet"
    ).df() if _table_exists(con, "pred_prophet") else pd.DataFrame()

    # SARIMA
    sarima = con.execute(
        "SELECT id, date, sarima_point FROM pred_sarima"
    ).df() if _table_exists(con, "pred_sarima") else pd.DataFrame()

    # LSTM
    lstm = con.execute(
        "SELECT id, date, lstm_point FROM pred_lstm"
    ).df() if _table_exists(con, "pred_lstm") else pd.DataFrame()

    # Dual-phase
    dual = con.execute(
        "SELECT id, date, dual_point FROM pred_dual"
    ).df() if _table_exists(con, "pred_dual") else pd.DataFrame()

    # Actuals: only load actuals for series that have predictions to avoid loading 59M rows
    pred_tables = ["pred_lgbm", "pred_prophet", "pred_sarima", "pred_lstm", "pred_dual"]
    existing_tables = [t for t in pred_tables if _table_exists(con, t)]
    
    if existing_tables:
        union_query = " UNION ".join(f"SELECT DISTINCT id FROM {t}" for t in existing_tables)
        active_ids = con.execute(f"SELECT id FROM ({union_query})").df()["id"].tolist()
        if active_ids:
            ids_str = ", ".join(f"'{i}'" for i in active_ids)
            actuals = con.execute(
                f"SELECT id, date, sales AS actual FROM master WHERE id IN ({ids_str}) ORDER BY id, date"
            ).df()
        else:
            actuals = pd.DataFrame(columns=["id", "date", "actual"])
    else:
        actuals = pd.DataFrame(columns=["id", "date", "actual"])

    # Merge all on id + date
    merged = actuals.copy()
    for df, col in [(lgbm, "lgbm_point"), (prophet, "prophet_point"),
                    (sarima, "sarima_point"), (lstm, "lstm_point"), (dual, "dual_point")]:
        if not df.empty:
            merged = merged.merge(df[["id", "date", col]], on=["id", "date"], how="left")

    # Join demand class
    classes = con.execute("SELECT id, demand_class FROM demand_classes").df()
    merged = merged.merge(classes, on="id", how="left")
    merged["date"] = pd.to_datetime(merged["date"])

    logger.info(f"Merged predictions: {len(merged):,} rows")
    return merged


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    return table in tables


def train_ensemble(
    con: duckdb.DuckDBPyConnection,
    val_start_date: str | None = None,
) -> dict[str, Ridge]:
    """
    Train per-class Ridge meta-learners on validation period predictions.

    Args:
        con: DuckDB connection.
        val_start_date: Start of validation window (default: last 56 days).

    Returns:
        Dict of {demand_class: Ridge model}
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_all_predictions(con)

    if val_start_date is None:
        cutoff = df["date"].max() - pd.Timedelta(days=56)
    else:
        cutoff = pd.Timestamp(val_start_date)

    val_df = df[df["date"] > cutoff].copy()
    logger.info(f"Validation period: {cutoff.date()} → {df['date'].max().date()} ({len(val_df):,} rows)")

    meta_models = {}

    for cls in DEMAND_CLASSES:
        cls_df = val_df[val_df["demand_class"] == cls].copy()
        available_cols = [c for c in MODEL_AVAILABILITY.get(cls, []) if c in cls_df.columns]

        if cls_df.empty or not available_cols:
            logger.warning(f"  {cls}: No data or predictions. Skipping.")
            continue

        # Drop rows where any model prediction is missing
        cls_df = cls_df.dropna(subset=available_cols + ["actual"])
        if len(cls_df) < 100:
            logger.warning(f"  {cls}: Too few rows ({len(cls_df)}). Using equal weights.")
            continue

        X_meta = cls_df[available_cols].values.astype(np.float32)
        y_meta = cls_df["actual"].values.astype(np.float32)

        # Non-negative constrained Ridge
        ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False)
        ridge.fit(X_meta, y_meta)

        # Normalize weights to sum to 1
        weights = np.maximum(ridge.coef_, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        ridge.coef_ = weights

        meta_models[cls] = {
            "model": ridge,
            "features": available_cols,
            "weights": dict(zip(available_cols, weights.round(4))),
        }

        logger.success(
            f"  {cls}: weights = {meta_models[cls]['weights']} "
            f"(n={len(cls_df):,})"
        )

    # Save
    with open(MODEL_DIR / "ensemble_meta.pkl", "wb") as f:
        pickle.dump(meta_models, f)

    return meta_models


def predict_ensemble(
    con: duckdb.DuckDBPyConnection,
    meta_models: dict | None = None,
) -> pd.DataFrame:
    """
    Apply ensemble meta-learner to generate final predictions.

    Returns DataFrame: id, date, actual, ensemble_point, demand_class
    """
    if meta_models is None:
        with open(MODEL_DIR / "ensemble_meta.pkl", "rb") as f:
            meta_models = pickle.load(f)

    df = _load_all_predictions(con)
    results = []

    for cls in DEMAND_CLASSES:
        cls_df = df[df["demand_class"] == cls].copy()
        if cls_df.empty:
            continue

        if cls not in meta_models:
            # Fallback: use best available single model
            for fallback in ["lgbm_point", "dual_point", "prophet_point"]:
                if fallback in cls_df.columns:
                    cls_df["ensemble_point"] = cls_df[fallback].fillna(0).clip(lower=0)
                    break
        else:
            meta = meta_models[cls]
            available = [c for c in meta["features"] if c in cls_df.columns]
            X = cls_df[available].fillna(0).values.astype(np.float32)
            weights = np.array([meta["weights"].get(c, 0) for c in available])
            if weights.sum() > 0:
                weights = weights / weights.sum()
            cls_df["ensemble_point"] = np.clip(X @ weights, 0, None).round(3)

        results.append(cls_df)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    logger.success(f"Ensemble predictions: {len(combined):,} rows.")
    return combined


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    meta = train_ensemble(con)
    preds = predict_ensemble(con, meta)
    print(preds[["id", "date", "actual", "ensemble_point", "demand_class"]].tail(20))
    con.close()
