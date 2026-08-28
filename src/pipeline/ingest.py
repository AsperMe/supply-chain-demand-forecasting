"""
Supply Chain Demand Forecasting — Data Ingestion
Loads raw M5 CSVs into a DuckDB analytical database.

Expected files in data/raw/:
  - sales_train_evaluation.csv
  - calendar.csv
  - sell_prices.csv
  (optional) sample_submission.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"


def _check_files() -> bool:
    """Verify required M5 raw files exist."""
    required = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
    missing = [f for f in required if not (RAW_DIR / f).exists()]
    if missing:
        logger.error(f"Missing raw files: {missing}")
        logger.info(f"Expected location: {RAW_DIR}")
        return False
    return True


def _load_sales(con: duckdb.DuckDBPyConnection) -> None:
    """Load wide-format sales → long format into DuckDB."""
    logger.info("Loading sales data (wide → long melt)…")
    path = RAW_DIR / "sales_train_evaluation.csv"

    # Read metadata columns + day columns
    df = pd.read_csv(path)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]

    # Melt to long format
    logger.info(f"  Melting {len(df):,} series × {len(day_cols)} days…")
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    df_long["sales"] = df_long["sales"].fillna(0).astype("int32")

    logger.info(f"  Long format rows: {len(df_long):,}")
    con.execute("DROP TABLE IF EXISTS raw_sales")
    con.execute("""
        CREATE TABLE raw_sales AS
        SELECT * FROM df_long
    """)
    logger.success("  raw_sales table created.")


def _load_calendar(con: duckdb.DuckDBPyConnection) -> None:
    """Load calendar with event and SNAP flags."""
    logger.info("Loading calendar data…")
    df = pd.read_csv(RAW_DIR / "calendar.csv")
    df["date"] = pd.to_datetime(df["date"])
    con.execute("DROP TABLE IF EXISTS raw_calendar")
    con.execute("CREATE TABLE raw_calendar AS SELECT * FROM df")
    logger.success(f"  raw_calendar: {len(df)} days loaded.")


def _load_prices(con: duckdb.DuckDBPyConnection) -> None:
    """Load sell prices per item/store/week."""
    logger.info("Loading sell prices…")
    df = pd.read_csv(RAW_DIR / "sell_prices.csv")
    con.execute("DROP TABLE IF EXISTS raw_prices")
    con.execute("CREATE TABLE raw_prices AS SELECT * FROM df")
    logger.success(f"  raw_prices: {len(df):,} rows loaded.")


def _build_master_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    Join sales + calendar + prices into a single analytical table.
    Schema: id, item_id, dept_id, cat_id, store_id, state_id,
            date, d, wm_yr_wk, sales, sell_price,
            weekday, month, year, event_name_1, event_type_1,
            snap_CA, snap_TX, snap_WI
    """
    logger.info("Building master analytical table…")
    con.execute("DROP TABLE IF EXISTS master")
    con.execute("""
        CREATE TABLE master AS
        SELECT
            s.id,
            s.item_id,
            s.dept_id,
            s.cat_id,
            s.store_id,
            s.state_id,
            c.date::DATE                          AS date,
            s.d,
            c.wm_yr_wk,
            CAST(s.sales AS INTEGER)              AS sales,
            COALESCE(p.sell_price, 0.0)           AS sell_price,
            c.weekday,
            EXTRACT(month FROM c.date)            AS month,
            EXTRACT(year  FROM c.date)            AS year,
            COALESCE(c.event_name_1, '')          AS event_name_1,
            COALESCE(c.event_type_1, '')          AS event_type_1,
            COALESCE(c.event_name_2, '')          AS event_name_2,
            COALESCE(c.event_type_2, '')          AS event_type_2,
            CAST(c.snap_CA AS INTEGER)            AS snap_CA,
            CAST(c.snap_TX AS INTEGER)            AS snap_TX,
            CAST(c.snap_WI AS INTEGER)            AS snap_WI
        FROM raw_sales s
        JOIN raw_calendar c
            ON s.d = c.d
        LEFT JOIN raw_prices p
            ON  p.store_id  = s.store_id
            AND p.item_id   = s.item_id
            AND p.wm_yr_wk  = c.wm_yr_wk
        ORDER BY s.id, c.date
    """)
    count = con.execute("SELECT COUNT(*) FROM master").fetchone()[0]
    logger.success(f"  master table: {count:,} rows.")


def ingest(force: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Run full ingestion pipeline. Returns open DuckDB connection.
    Set force=True to re-ingest even if DB already exists.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists() and not force:
        logger.info(f"DuckDB already exists at {DB_PATH}. Use force=True to re-ingest.")
        return duckdb.connect(str(DB_PATH))

    if not _check_files():
        sys.exit(1)

    logger.info(f"Connecting to DuckDB at {DB_PATH}…")
    con = duckdb.connect(str(DB_PATH))

    _load_sales(con)
    _load_calendar(con)
    _load_prices(con)
    _build_master_table(con)

    logger.success("Ingestion complete!")
    return con


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest M5 raw data into DuckDB")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if DB exists")
    args = parser.parse_args()

    con = ingest(force=args.force)
    print("\nTables in DB:")
    for row in con.execute("SHOW TABLES").fetchall():
        tbl = row[0]
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {n:,} rows")
    con.close()
