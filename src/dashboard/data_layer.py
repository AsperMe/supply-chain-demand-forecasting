"""
Dashboard Data Layer — loads and caches data from DuckDB for the dashboard.
Optimized for dynamic, on-demand querying to prevent high memory usage.
"""

from __future__ import annotations

import functools
from pathlib import Path
# pyrefly: ignore [missing-import]
import duckdb
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection | None:
    """Return read-only DuckDB connection (returns None if missing)."""
    if not DB_PATH.exists():
        logger.warning(f"DuckDB not found at {DB_PATH}. Using demo data.")
        return None
    return duckdb.connect(str(DB_PATH), read_only=True)


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    return table in tables


@functools.lru_cache(maxsize=1)
def get_prediction_date_range() -> tuple[str, str]:
    """Return the min and max dates available in the predictions."""
    con = get_connection()
    if con is None or not _table_exists(con, "pred_ensemble"):
        return "2016-01-01", "2016-04-30"
    try:
        row = con.execute("SELECT MIN(date), MAX(date) FROM pred_ensemble").fetchone()
        con.close()
        min_date = row[0].strftime("%Y-%m-%d") if row[0] else "2016-01-01"
        max_date = row[1].strftime("%Y-%m-%d") if row[1] else "2016-04-30"
        return min_date, max_date
    except Exception as e:
        logger.error(f"Error getting date range: {e}")
        return "2016-01-01", "2016-04-30"


@functools.lru_cache(maxsize=1)
def load_demand_classes() -> pd.DataFrame:
    """Return demand classes metadata (loaded once)."""
    con = get_connection()
    if con is None or not _table_exists(con, "demand_classes"):
        return _demo_demand_classes()
    try:
        df = con.execute("SELECT * FROM demand_classes").df()
        con.close()
        return df
    except Exception as e:
        logger.error(f"Error loading demand classes: {e}")
        return _demo_demand_classes()


@functools.lru_cache(maxsize=1)
def get_filter_options() -> dict:
    """Return unique dropdown options for State, Store, and Category."""
    classes = load_demand_classes()
    return {
        "states": sorted(classes["state_id"].dropna().unique().tolist()),
        "stores": sorted(classes["store_id"].dropna().unique().tolist()),
        "categories": sorted(classes["cat_id"].dropna().unique().tolist()),
        "demand_classes": ["Smooth", "Intermittent", "Erratic", "Lumpy"],
    }


def get_filtered_items(
    state_id: str | None = None,
    store_id: str | None = None,
    cat_id: str | None = None,
) -> list[dict]:
    """Return list of SKU dicts (id, demand_class) matching the selected filters."""
    con = get_connection()
    if con is None or not _table_exists(con, "demand_classes"):
        return _demo_filtered_items(state_id, store_id, cat_id)
    try:
        clauses = []
        params = []
        if state_id:
            clauses.append("state_id = ?")
            params.append(state_id)
        if store_id:
            clauses.append("store_id = ?")
            params.append(store_id)
        if cat_id:
            clauses.append("cat_id = ?")
            params.append(cat_id)

        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"SELECT id, demand_class FROM demand_classes {where_sql} ORDER BY id LIMIT 200"
        df = con.execute(query, params).df()
        con.close()
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"Error loading filtered items: {e}")
        return _demo_filtered_items(state_id, store_id, cat_id)


def get_overview_kpis(
    state_id: str | None = None,
    store_id: str | None = None,
    cat_id: str | None = None,
) -> dict:
    """Compute and return aggregated overview KPIs for the last 28 days."""
    con = get_connection()
    if con is None or not _table_exists(con, "pred_ensemble"):
        return _demo_kpis()

    try:
        # Build SQL where filters
        clauses = []
        params = []
        if state_id:
            clauses.append("dc.state_id = ?")
            params.append(state_id)
        if store_id:
            clauses.append("dc.store_id = ?")
            params.append(store_id)
        if cat_id:
            clauses.append("dc.cat_id = ?")
            params.append(cat_id)

        where_sql = " AND ".join(clauses) if clauses else "1=1"

        # Query DuckDB
        query = f"""
            WITH latest_date AS (
                SELECT MAX(date) AS max_date FROM pred_ensemble
            ),
            recent_data AS (
                SELECT p.id, p.actual, p.ensemble_point, l.lgbm_q10, l.lgbm_q90
                FROM pred_ensemble p
                LEFT JOIN demand_classes dc ON p.id = dc.id
                LEFT JOIN pred_lgbm l ON p.id = l.id AND p.date = l.date
                WHERE p.date >= (SELECT max_date - INTERVAL '28 days' FROM latest_date)
                  AND {where_sql}
            )
            SELECT
                -- WAPE-based Accuracy: 100 * (1 - sum(|actual - pred|) / sum(actual))
                GREATEST(0.0, ROUND(100.0 * (1.0 - SUM(ABS(actual - ensemble_point)) / (SUM(actual) + 1e-10)), 1)) AS accuracy,
                -- Safety Stockout Alerts (q10 is 0 while forecast is positive)
                COUNT(DISTINCT CASE WHEN lgbm_q10 = 0 AND ensemble_point > 1 THEN id END) AS stockout_alerts,
                -- Overstock Alerts (upper bounds significantly exceed point forecast)
                COUNT(DISTINCT CASE WHEN lgbm_q90 > 3.0 * ensemble_point AND ensemble_point > 0.0 THEN id END) AS overstock_alerts,
                -- Sales Volume
                COALESCE(SUM(actual), 0) AS total_sales
            FROM recent_data
        """
        row = con.execute(query, params).fetchone()
        con.close()

        stockouts = int(row[1]) if row[1] is not None else 0
        overstocks = int(row[2]) if row[2] is not None else 0
        
        return {
            "accuracy": float(row[0]) if row[0] is not None else 85.0,
            "stockout_alerts": stockouts,
            "overstock_alerts": overstocks,
            "total_sales": int(row[3]) if row[3] is not None else 0,
            "revenue_at_risk": stockouts * 150,  # Proxy: $150 average per stockout alert
        }
    except Exception as e:
        logger.error(f"Error computing KPIs: {e}")
        return _demo_kpis()


def get_weekly_trend(
    state_id: str | None = None,
    store_id: str | None = None,
    cat_id: str | None = None,
) -> pd.DataFrame:
    """Return weekly aggregated actual vs forecast sales trend."""
    con = get_connection()
    if con is None or not _table_exists(con, "pred_ensemble"):
        return _demo_weekly_trend()

    try:
        clauses = []
        params = []
        if state_id:
            clauses.append("dc.state_id = ?")
            params.append(state_id)
        if store_id:
            clauses.append("dc.store_id = ?")
            params.append(store_id)
        if cat_id:
            clauses.append("dc.cat_id = ?")
            params.append(cat_id)

        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""

        query = f"""
            SELECT
                date_trunc('week', p.date) AS date,
                SUM(p.actual) AS actual,
                SUM(p.ensemble_point) AS forecast
            FROM pred_ensemble p
            LEFT JOIN demand_classes dc ON p.id = dc.id
            {where_sql}
            GROUP BY 1
            ORDER BY 1
        """
        df = con.execute(query, params).df()
        con.close()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.error(f"Error loading weekly trend: {e}")
        return _demo_weekly_trend()


def get_demand_class_counts(
    state_id: str | None = None,
    store_id: str | None = None,
    cat_id: str | None = None,
) -> pd.DataFrame:
    """Return SKU counts per demand class."""
    con = get_connection()
    if con is None or not _table_exists(con, "demand_classes"):
        return _demo_demand_class_counts()

    try:
        clauses = []
        params = []
        if state_id:
            clauses.append("state_id = ?")
            params.append(state_id)
        if store_id:
            clauses.append("store_id = ?")
            params.append(store_id)
        if cat_id:
            clauses.append("cat_id = ?")
            params.append(cat_id)

        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""

        query = f"""
            SELECT demand_class AS class, COUNT(*) AS count
            FROM demand_classes
            {where_sql}
            GROUP BY demand_class
        """
        df = con.execute(query, params).df()
        con.close()
        return df
    except Exception as e:
        logger.error(f"Error loading demand class counts: {e}")
        return _demo_demand_class_counts()


def get_forecast_data(
    item_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Return daily actuals and predictions for a single selected SKU."""
    con = get_connection()
    if con is None or not _table_exists(con, "pred_ensemble"):
        return _demo_forecast_data(item_id, start_date, end_date)

    try:
        query = """
            SELECT
                p.date,
                p.actual,
                p.ensemble_point,
                p.lgbm_point,
                p.prophet_point,
                p.sarima_point,
                p.lstm_point,
                p.dual_point,
                l.lgbm_q10,
                l.lgbm_q90
            FROM pred_ensemble p
            LEFT JOIN pred_lgbm l ON p.id = l.id AND p.date = l.date
            WHERE p.id = ? AND p.date >= ? AND p.date <= ?
            ORDER BY p.date
        """
        df = con.execute(query, [item_id, start_date, end_date]).df()
        con.close()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.error(f"Error loading forecast data for {item_id}: {e}")
        return _demo_forecast_data(item_id, start_date, end_date)


# ── Demo Data Generators ──────────────────────────────────────────────────────

def _demo_demand_classes() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    classes = np.random.choice(["Smooth", "Intermittent", "Erratic", "Lumpy"], size=n, p=[0.45, 0.25, 0.18, 0.12])
    states = np.random.choice(["CA", "TX", "WI"], size=n)
    stores = [f"{s}_{np.random.randint(1, 4)}" for s in states]
    cats = np.random.choice(["HOBBIES", "HOUSEHOLD", "FOODS"], size=n)
    depts = [f"{c}_{np.random.randint(1, 3)}" for c in cats]
    items = [f"{d}_{i:03d}" for i, d in enumerate(depts)]
    ids = [f"{item}_{store}" for item, store in zip(items, stores)]

    return pd.DataFrame({
        "id": ids,
        "item_id": items,
        "dept_id": depts,
        "cat_id": cats,
        "store_id": stores,
        "state_id": states,
        "demand_class": classes,
    })


def _demo_filtered_items(state=None, store=None, cat=None) -> list[dict]:
    df = _demo_demand_classes()
    mask = pd.Series([True] * len(df))
    if state: mask &= df["state_id"] == state
    if store: mask &= df["store_id"] == store
    if cat:   mask &= df["cat_id"] == cat
    subset = df[mask][["id", "demand_class"]].head(100)
    return subset.to_dict("records")


def _demo_kpis() -> dict:
    return {
        "accuracy": 83.2,
        "stockout_alerts": 42,
        "overstock_alerts": 18,
        "total_sales": 15480,
        "revenue_at_risk": 6300,
    }


def _demo_weekly_trend() -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", "2016-04-30", freq="W")
    np.random.seed(42)
    base = 12000
    actual = base + np.random.normal(0, 1000, len(dates))
    forecast = actual * np.random.uniform(0.9, 1.1)
    return pd.DataFrame({"date": dates, "actual": actual, "forecast": forecast})


def _demo_demand_class_counts() -> pd.DataFrame:
    return pd.DataFrame({
        "class": ["Smooth", "Intermittent", "Erratic", "Lumpy"],
        "count": [90, 50, 36, 24]
    })


def _demo_forecast_data(item_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date or "2016-01-01")
    end = pd.Timestamp(end_date or "2016-04-30")
    dates = pd.date_range(start, end, freq="D")
    
    np.random.seed(42)
    base = np.random.uniform(1.0, 10.0)
    actual = np.clip(base + np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 2 + np.random.normal(0, 1, len(dates)), 0, None).round(0)
    ensemble = actual * np.random.uniform(0.92, 1.08)
    lgbm = actual * np.random.uniform(0.90, 1.10)
    
    return pd.DataFrame({
        "date": dates,
        "actual": actual,
        "ensemble_point": ensemble,
        "lgbm_point": lgbm,
        "prophet_point": ensemble * 0.95,
        "sarima_point": ensemble * 1.05,
        "lstm_point": ensemble * 0.98,
        "dual_point": ensemble,
        "lgbm_q10": np.clip(ensemble * 0.6, 0, None),
        "lgbm_q90": ensemble * 1.4,
    })
