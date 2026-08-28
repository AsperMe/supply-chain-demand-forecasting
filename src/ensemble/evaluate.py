"""
Supply Chain Demand Forecasting — Evaluation Metrics
RMSE, MAE, and WRMSSE (the official M5 competition metric).

WRMSSE = Weighted Root Mean Squared Scaled Error
  - Weights proportional to last 28-day revenue (sell_price × sales)
  - Scaled by in-sample naive forecast error (denominator)
  - Hierarchical: computed across all levels (item, store, state, national)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"
REPORTS_DIR = ROOT / "reports"


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    errors = np.zeros_like(denom, dtype=float)
    np.divide(np.abs(actual - predicted), denom, out=errors, where=denom != 0)
    return float(np.mean(errors) * 100)


def _compute_scaling_factor(series: np.ndarray) -> float:
    """
    RMSSE denominator: RMSE of naive (lag-1) forecast on training data.
    Avoids dividing by zero for flat series.
    """
    diffs = np.diff(series)
    if len(diffs) == 0 or np.all(diffs == 0):
        return 1.0
    return float(np.sqrt(np.mean(diffs ** 2)))


def wrmsse(
    predictions_df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    pred_col: str = "ensemble_point",
    val_days: int = 28,
) -> float:
    """
    Compute WRMSSE over the validation horizon.

    Args:
        predictions_df: DataFrame with id, date, actual, <pred_col>
        con: DuckDB connection (for training history and weights)
        pred_col: Column name of predictions
        val_days: Length of validation horizon

    Returns:
        WRMSSE score (lower is better)
    """
    logger.info(f"Computing WRMSSE (pred_col={pred_col}, horizon={val_days})…")

    # Revenue weights: last 28 days of training data
    weight_df = con.execute(f"""
        SELECT id, SUM(sales * sell_price) AS revenue
        FROM master
        WHERE date >= (SELECT MAX(date) - INTERVAL '{val_days + 28} days' FROM master)
          AND date <= (SELECT MAX(date) - INTERVAL '{val_days} days' FROM master)
        GROUP BY id
    """).df()
    total_revenue = weight_df["revenue"].sum()
    weight_df["weight"] = weight_df["revenue"] / (total_revenue + 1e-10)
    weight_map = weight_df.set_index("id")["weight"].to_dict()

    # Training data for scaling factors
    train_end = pd.to_datetime(predictions_df["date"].max()) - pd.Timedelta(days=val_days)
    all_series = predictions_df["id"].unique()

    ids_str = ", ".join(f"'{i}'" for i in all_series)
    train_df = con.execute(
        f"SELECT id, date, sales FROM master WHERE id IN ({ids_str}) "
        f"AND date <= '{train_end.date()}' ORDER BY id, date"
    ).df()
    scale_map = (
        train_df.groupby("id")["sales"]
        .apply(lambda s: _compute_scaling_factor(s.values.astype(float)))
        .to_dict()
    )

    # Validation predictions
    val_cutoff = predictions_df["date"].max() - pd.Timedelta(days=val_days)
    val_df = predictions_df[predictions_df["date"] > val_cutoff].copy()

    if pred_col not in val_df.columns:
        logger.error(f"Column {pred_col} not found in predictions_df")
        return float("nan")

    rmsse_scores = []
    weights_used = []

    for series_id, group in val_df.groupby("id"):
        actual = group["actual"].values.astype(float)
        predicted = group[pred_col].fillna(0).values.astype(float)

        if len(actual) == 0:
            continue

        # Scale factor from training data
        scale = scale_map.get(series_id, 1.0)

        # RMSSE for this series
        num = np.sqrt(np.mean((actual - predicted) ** 2))
        series_rmsse = num / (scale + 1e-10)

        w = weight_map.get(series_id, 0.0)
        rmsse_scores.append(series_rmsse)
        weights_used.append(w)

    if not rmsse_scores:
        return float("nan")

    weights_arr = np.array(weights_used)
    if weights_arr.sum() > 0:
        weights_arr = weights_arr / weights_arr.sum()
    else:
        weights_arr = np.ones(len(weights_used)) / len(weights_used)

    score = float(np.dot(weights_arr, rmsse_scores))
    logger.success(f"WRMSSE = {score:.4f}")
    return score


def evaluate_all_models(
    predictions_df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    val_days: int = 28,
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, SMAPE, WRMSSE for all model columns in predictions_df.

    Returns:
        DataFrame with model as index, metrics as columns
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_cols = [c for c in predictions_df.columns
                  if c.endswith("_point") or c in ["dual_point", "ensemble_point"]]

    val_cutoff = pd.to_datetime(predictions_df["date"]).max() - pd.Timedelta(days=val_days)
    val_df = predictions_df[pd.to_datetime(predictions_df["date"]) > val_cutoff].copy()
    val_df = val_df.dropna(subset=["actual"])

    rows = []
    for col in model_cols:
        if col not in val_df.columns:
            continue
        sub = val_df.dropna(subset=[col])
        if sub.empty:
            continue
        actual = sub["actual"].values.astype(float)
        pred = sub[col].values.astype(float)

        row = {
            "model": col.replace("_point", "").replace("_", " ").title(),
            "col": col,
            "n_series": sub["id"].nunique(),
            "n_rows": len(sub),
            "RMSE": round(rmse(actual, pred), 4),
            "MAE": round(mae(actual, pred), 4),
            "SMAPE": round(smape(actual, pred), 4),
        }
        # WRMSSE (slower, only for main models)
        if col in ["lgbm_point", "ensemble_point", "dual_point"]:
            row["WRMSSE"] = round(wrmsse(
                val_df[["id", "date", "actual", col]].rename(columns={col: col}),
                con, pred_col=col, val_days=val_days
            ), 4)
        else:
            row["WRMSSE"] = None

        rows.append(row)

    results = pd.DataFrame(rows).set_index("model")
    logger.success("Evaluation complete:")
    print(results.to_string())

    results.to_csv(REPORTS_DIR / "model_comparison.csv")
    logger.info(f"Saved: {REPORTS_DIR / 'model_comparison.csv'}")
    return results


def compute_per_class_metrics(
    predictions_df: pd.DataFrame,
    model_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute RMSE and MAE broken down by demand class."""
    if model_cols is None:
        model_cols = [c for c in predictions_df.columns
                      if c.endswith("_point") or c == "ensemble_point"]

    rows = []
    for cls in predictions_df["demand_class"].dropna().unique():
        cls_df = predictions_df[predictions_df["demand_class"] == cls]
        for col in model_cols:
            if col not in cls_df.columns:
                continue
            sub = cls_df.dropna(subset=[col, "actual"])
            if sub.empty:
                continue
            actual = sub["actual"].values.astype(float)
            pred = sub[col].values.astype(float)
            rows.append({
                "demand_class": cls,
                "model": col,
                "RMSE": round(rmse(actual, pred), 4),
                "MAE": round(mae(actual, pred), 4),
                "n": len(sub),
            })

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "per_class_metrics.csv", index=False)
    return df


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    if _table := "pred_lgbm":
        from src.ensemble.stacker import predict_ensemble
        preds = predict_ensemble(con)
        if not preds.empty:
            evaluate_all_models(preds, con)
    con.close()
