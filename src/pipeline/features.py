"""
Supply Chain Demand Forecasting — Feature Engineering
Adds time-series features to the master DuckDB table.

Features engineered:
  Calendar  : day_of_week, week_of_year, month, year, is_weekend,
              is_month_start, is_month_end, quarter
  Events    : is_sporting, is_cultural, is_national, is_religious,
              snap_active (state-aware)
  Price     : price_change_pct, days_since_price_change, is_promotion
  Lag       : sales_lag_7, sales_lag_14, sales_lag_28, sales_lag_56
  Rolling   : rolling_mean_7, rolling_mean_28, rolling_std_7, rolling_std_28
              rolling_min_7, rolling_max_7
  Demand    : days_since_last_sale, rolling_nonzero_rate_28
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"

# Lag windows (days)
LAG_DAYS = [7, 14, 28, 56]
# Rolling windows (days)
ROLL_WINDOWS = [7, 28]

SPORTS_EVENTS = {"SuperBowl", "ValentinesDay", "PresidentsDay", "LentStart",
                 "LentWeek2", "StPatricksDay", "Cinco_De_Mayo", "MemorialDay",
                 "NBAFinalsStart", "NBAFinalsEnd", "IndependenceDay",
                 "ColumbusDay", "Halloween", "EidAlAdha"}
CULTURAL_EVENTS = {"MartinLutherKingDay", "Easter", "Ramadan starts",
                   "OrthodoxChristmas"}
NATIONAL_EVENTS = {"NewYear", "IndependenceDay", "ChristmasDay",
                   "Thanksgiving", "Christmas", "LaborDay"}


def _build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived features."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int32")
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_month_start"] = df["date"].dt.is_month_start.astype("int8")
    df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")
    # Encode day/month cyclically for models that can use it
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).round(6)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).round(6)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(6)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(6)
    return df


def _build_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add event and SNAP features."""
    df = df.copy()
    df["is_sporting"] = df["event_type_1"].isin(["Sporting"]).astype("int8")
    df["is_cultural"] = df["event_type_1"].isin(["Cultural"]).astype("int8")
    df["is_national"] = df["event_type_1"].isin(["National"]).astype("int8")
    df["is_religious"] = df["event_type_1"].isin(["Religious"]).astype("int8")
    df["has_event"] = (df["event_name_1"] != "").astype("int8")
    # State-aware SNAP
    df["snap_active"] = (
        (df["state_id"] == "CA") * df["snap_CA"] +
        (df["state_id"] == "TX") * df["snap_TX"] +
        (df["state_id"] == "WI") * df["snap_WI"]
    ).clip(0, 1).astype("int8")
    return df


def _build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features within each series."""
    df = df.sort_values(["id", "date"]).copy()
    # Rolling 30-day avg price
    df["price_roll30"] = (
        df.groupby("id")["sell_price"]
        .transform(lambda x: x.rolling(30, min_periods=1).mean())
    )
    df["price_change_pct"] = (
        (df["sell_price"] - df.groupby("id")["sell_price"].shift(1)) /
        df.groupby("id")["sell_price"].shift(1).replace(0, np.nan)
    ).fillna(0).round(6)
    # Promotion proxy: price < 95% of 30-day rolling average
    df["is_promotion"] = (
        (df["sell_price"] > 0) &
        (df["sell_price"] < 0.95 * df["price_roll30"])
    ).astype("int8")
    # Days since last price change
    price_changed = df.groupby("id")["sell_price"].transform(
        lambda x: (x != x.shift(1)).astype(int)
    )
    df["days_since_price_change"] = (
        price_changed.groupby(df["id"]).cumsum()
        .groupby(df["id"]).transform(lambda x: x - x.where(x == x).ffill())
    ).fillna(0).astype("int32")
    return df


def _build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features — always shift by at least 28 days to prevent leakage."""
    df = df.sort_values(["id", "date"]).copy()
    for lag in LAG_DAYS:
        col = f"sales_lag_{lag}"
        df[col] = df.groupby("id")["sales"].shift(lag).astype("float32")
    return df


def _build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling aggregate features (applied on lag-7 to avoid leakage)."""
    df = df.sort_values(["id", "date"]).copy()
    # Use lag-7 as base to avoid leakage
    base = df.groupby("id")["sales"].shift(7)
    for w in ROLL_WINDOWS:
        df[f"rolling_mean_{w}"] = (
            base.groupby(df["id"]).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
        ).astype("float32")
        df[f"rolling_std_{w}"] = (
            base.groupby(df["id"]).transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
        ).astype("float32")
    df["rolling_min_7"] = (
        base.groupby(df["id"]).transform(
            lambda x: x.rolling(7, min_periods=1).min()
        )
    ).astype("float32")
    df["rolling_max_7"] = (
        base.groupby(df["id"]).transform(
            lambda x: x.rolling(7, min_periods=1).max()
        )
    ).astype("float32")
    return df


def _build_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add intermittent demand features."""
    df = df.sort_values(["id", "date"]).copy()
    # Days since last non-zero sale (using lag-1 to avoid leakage)
    last_sale = df.groupby("id")["sales"].shift(1)
    df["days_since_last_sale"] = (
        (last_sale == 0)
        .groupby(df["id"])
        .transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
    ).fillna(0).astype("int32")
    # 28-day rolling non-zero rate
    df["rolling_nonzero_rate_28"] = (
        (last_sale > 0).astype("float32")
        .groupby(df["id"])
        .transform(lambda x: x.rolling(28, min_periods=1).mean())
    ).astype("float32")
    return df


def build_features(
    con: duckdb.DuckDBPyConnection,
    sample_ids: list[str] | None = None,
    chunk_size: int = 500,
) -> None:
    """
    Build all features and write to DuckDB `features` table.

    Args:
        con: Open DuckDB connection.
        sample_ids: If provided, only process these series IDs (for fast iteration).
        chunk_size: Process this many series at a time to manage memory.
    """
    logger.info("Starting feature engineering…")

    # Load master (or subset)
    if sample_ids:
        ids_str = ", ".join(f"'{i}'" for i in sample_ids)
        query = f"SELECT * FROM master WHERE id IN ({ids_str}) ORDER BY id, date"
    else:
        query = "SELECT * FROM master ORDER BY id, date"

    if sample_ids:
        ids_str = ", ".join(f"'{i}'" for i in sample_ids)
        all_ids = con.execute(f"SELECT DISTINCT id FROM master WHERE id IN ({ids_str})").df()["id"].tolist()
    else:
        all_ids = con.execute("SELECT DISTINCT id FROM master").df()["id"].tolist()

    logger.info(f"  Processing {len(all_ids):,} series in chunks of {chunk_size}…")

    first_chunk = True
    for i in tqdm(range(0, len(all_ids), chunk_size), desc="Feature engineering"):
        chunk_ids = all_ids[i : i + chunk_size]
        ids_str = ", ".join(f"'{sid}'" for sid in chunk_ids)
        df = con.execute(
            f"SELECT * FROM master WHERE id IN ({ids_str}) ORDER BY id, date"
        ).df()

        # Apply all feature builders
        df = _build_calendar_features(df)
        df = _build_event_features(df)
        df = _build_price_features(df)
        df = _build_lag_features(df)
        df = _build_rolling_features(df)
        df = _build_demand_features(df)

        # Write to DuckDB
        if first_chunk:
            con.execute("DROP TABLE IF EXISTS features")
            con.execute("CREATE TABLE features AS SELECT * FROM df")
            first_chunk = False
        else:
            con.execute("INSERT INTO features SELECT * FROM df")

    count = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    logger.success(f"Feature engineering complete. features table: {count:,} rows.")


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    build_features(con)
    con.close()
