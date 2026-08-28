"""
Supply Chain Demand Forecasting — Demand Classification
Classifies each SKU using the ADI × CV² grid (Syntetos-Boylan framework).

Classes:
  Smooth       — ADI < 1.32,  CV² < 0.49   (regular, predictable)
  Intermittent — ADI ≥ 1.32,  CV² < 0.49   (sparse but consistent size)
  Erratic      — ADI < 1.32,  CV² ≥ 0.49   (frequent but highly variable)
  Lumpy        — ADI ≥ 1.32,  CV² ≥ 0.49   (sparse AND highly variable)

These classes drive which model is applied:
  Smooth       → SARIMA + Prophet + LightGBM + LSTM → ensemble
  Intermittent → Dual-phase LightGBM (occurrence + magnitude)
  Erratic      → LightGBM + Prophet → ensemble
  Lumpy        → Dual-phase LightGBM (primary), ensemble as fallback
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"

# ADI and CV² thresholds (Syntetos-Boylan)
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def _compute_adi(series: pd.Series) -> float:
    """Average Demand Interval = total_periods / non_zero_periods."""
    n_nonzero = (series > 0).sum()
    if n_nonzero == 0:
        return float("inf")
    return len(series) / n_nonzero


def _compute_cv2(series: pd.Series) -> float:
    """Squared Coefficient of Variation of non-zero demand sizes."""
    nonzero = series[series > 0]
    if len(nonzero) < 2:
        return 0.0
    mean = nonzero.mean()
    if mean == 0:
        return 0.0
    return (nonzero.std() / mean) ** 2


def _classify(adi: float, cv2: float) -> str:
    """Map ADI and CV² to demand class."""
    if adi >= ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        return "Lumpy"
    elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "Intermittent"
    elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        return "Erratic"
    else:
        return "Smooth"


def classify_demand(
    con: duckdb.DuckDBPyConnection,
    history_days: int = 365,
) -> pd.DataFrame:
    """
    Compute ADI, CV², and demand class for every series.

    Args:
        con: Open DuckDB connection.
        history_days: Number of most-recent days to use for classification.

    Returns:
        DataFrame with columns: id, item_id, dept_id, cat_id, store_id,
        state_id, adi, cv2, demand_class, zero_rate, mean_nonzero_sales,
        total_periods, nonzero_periods
    """
    logger.info(f"Classifying demand (using last {history_days} days)…")

    query = f"""
        SELECT id, item_id, dept_id, cat_id, store_id, state_id, date, sales
        FROM master
        WHERE date >= (SELECT MAX(date) - INTERVAL '{history_days} days' FROM master)
        ORDER BY id, date
    """
    df = con.execute(query).df()

    results = []
    for series_id, group in df.groupby("id"):
        sales = group["sales"].values.astype(float)
        adi = _compute_adi(pd.Series(sales))
        cv2 = _compute_cv2(pd.Series(sales))
        demand_class = _classify(adi, cv2)
        meta = group.iloc[0][["item_id", "dept_id", "cat_id", "store_id", "state_id"]]
        results.append({
            "id": series_id,
            "item_id": meta["item_id"],
            "dept_id": meta["dept_id"],
            "cat_id": meta["cat_id"],
            "store_id": meta["store_id"],
            "state_id": meta["state_id"],
            "adi": round(adi, 4),
            "cv2": round(cv2, 4),
            "demand_class": demand_class,
            "zero_rate": round((sales == 0).mean(), 4),
            "mean_nonzero_sales": round(float(sales[sales > 0].mean()) if (sales > 0).any() else 0.0, 4),
            "total_periods": len(sales),
            "nonzero_periods": int((sales > 0).sum()),
        })

    classifications = pd.DataFrame(results)

    # Store in DuckDB
    con.execute("DROP TABLE IF EXISTS demand_classes")
    con.execute("CREATE TABLE demand_classes AS SELECT * FROM classifications")

    # Print summary
    dist = classifications["demand_class"].value_counts()
    logger.success("Demand classification complete:")
    for cls, count in dist.items():
        pct = 100 * count / len(classifications)
        logger.info(f"  {cls:15s}: {count:6,} series ({pct:.1f}%)")

    return classifications


def get_series_by_class(
    con: duckdb.DuckDBPyConnection,
    demand_class: str,
) -> list[str]:
    """Return list of series IDs for a given demand class."""
    result = con.execute(
        "SELECT id FROM demand_classes WHERE demand_class = ?",
        [demand_class],
    ).df()
    return result["id"].tolist()


def get_model_routing(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """
    Returns a dict mapping model name → list of series IDs it should handle.
    Routing logic:
      SARIMA   → Smooth only (sufficient history, low variability)
      Prophet  → Smooth + Erratic
      LightGBM → All classes
      LSTM     → Smooth + Erratic
      DualPhase→ Intermittent + Lumpy
    """
    smooth = get_series_by_class(con, "Smooth")
    intermittent = get_series_by_class(con, "Intermittent")
    erratic = get_series_by_class(con, "Erratic")
    lumpy = get_series_by_class(con, "Lumpy")

    return {
        "sarima": smooth,
        "prophet": smooth + erratic,
        "lgbm": smooth + intermittent + erratic + lumpy,
        "lstm": smooth + erratic,
        "dual_phase": intermittent + lumpy,
    }


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    df = classify_demand(con)
    print(df.head(10).to_string())
    routing = get_model_routing(con)
    print("\nModel routing:")
    for model, ids in routing.items():
        print(f"  {model}: {len(ids):,} series")
    con.close()
