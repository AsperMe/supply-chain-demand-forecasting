from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"
MODEL_DIR = ROOT / "data" / "models"

# US public holidays that affect retail
US_HOLIDAYS = pd.DataFrame({
    "holiday": [
        "NewYear", "MLK", "PresidentsDay", "Easter", "MemorialDay",
        "IndependenceDay", "LaborDay", "Thanksgiving", "Christmas",
        "SuperBowl", "ValentinesDay", "Halloween",
    ],
    "ds": pd.to_datetime([
        "2011-01-01", "2011-01-17", "2011-02-21", "2011-04-24", "2011-05-30",
        "2011-07-04", "2011-09-05", "2011-11-24", "2011-12-25",
        "2011-02-06", "2011-02-14", "2011-10-31",
    ]),
    "lower_window": [-1, 0, 0, 0, -1, -1, -1, -2, -3, -1, -1, -1],
    "upper_window": [1, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0],
})


def _get_series_df(
    con: duckdb.DuckDBPyConnection,
    series_id: str,
) -> pd.DataFrame:
    """Load a single time series from DuckDB in Prophet format (ds, y)."""
    df = con.execute(
        "SELECT date AS ds, sales AS y FROM master WHERE id = ? ORDER BY ds",
        [series_id],
    ).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def train_prophet_single(
    series_df: pd.DataFrame,
    series_id: str,
    horizon: int = 28,
) -> dict:
    """
    Train Prophet on a single series and generate in-sample + horizon forecast.

    Returns dict with keys: id, train_preds, forecast
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Install prophet: pip install prophet")

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=US_HOLIDAYS,
        uncertainty_samples=200,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(series_df)

    # In-sample predictions
    train_future = m.make_future_dataframe(periods=0)
    train_forecast = m.predict(train_future)

    # Out-of-sample forecast
    future = m.make_future_dataframe(periods=horizon, freq="D")
    forecast = m.predict(future)

    return {
        "id": series_id,
        "model": m,
        "train_preds": train_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={"yhat": "prophet_point", "yhat_lower": "prophet_lower", "yhat_upper": "prophet_upper"}
        ),
        "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).rename(
            columns={"ds": "date", "yhat": "prophet_point", "yhat_lower": "prophet_lower", "yhat_upper": "prophet_upper"}
        ),
        "trend": forecast[["ds", "trend"]].rename(columns={"ds": "date"}),
    }


def _prophet_worker(
    sid: str,
    db_file: str | None,
    horizon: int,
    is_forecast: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame | None:
    """Module-level worker function that is picklable."""
    try:
        # If in-memory, reuse the active connection; otherwise, open a new read-only connection
        if con is None:
            if not db_file:
                return None
            worker_con = duckdb.connect(db_file, read_only=True)
            df = _get_series_df(worker_con, sid)
            worker_con.close()
        else:
            df = _get_series_df(con, sid)

        if len(df) < 60:  # Need enough history
            return None
        result = train_prophet_single(df, sid, horizon)
        if is_forecast:
            fc = result["forecast"].copy()
            fc["id"] = sid
            return fc
        else:
            preds = result["train_preds"].copy()
            preds["id"] = sid
            preds = preds.rename(columns={"ds": "date"})
            return preds
    except Exception as e:
        logger.warning(f"  Prophet worker failed for {sid}: {e}")
        return None


def train_predict_prophet(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    horizon: int = 28,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    Train Prophet for multiple series and return combined predictions.

    Returns DataFrame: id, date, prophet_point, prophet_lower, prophet_upper
    """
    from joblib import Parallel, delayed

    logger.info(f"Training Prophet on {len(series_ids):,} series…")

    # Check if database is in-memory
    db_list = con.execute("PRAGMA database_list").df()
    db_file = db_list["file"].iloc[0]
    is_memory = db_file is None or db_file == ""

    if is_memory or n_jobs <= 1:
        # Run sequentially on the same connection (important for in-memory)
        results = [
            _prophet_worker(sid, None, horizon, is_forecast=False, con=con)
            for sid in tqdm(series_ids, desc="Prophet")
        ]
    else:
        # Run in parallel
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_prophet_worker)(sid, db_file, horizon, is_forecast=False)
            for sid in tqdm(series_ids, desc="Prophet")
        )

    combined = pd.concat([r for r in results if r is not None], ignore_index=True)
    if not combined.empty:
        combined[["prophet_point", "prophet_lower", "prophet_upper"]] = (
            combined[["prophet_point", "prophet_lower", "prophet_upper"]].clip(lower=0)
        )
        logger.success(f"Prophet predictions: {len(combined):,} rows across {combined['id'].nunique():,} series.")
    else:
        logger.warning("Prophet produced no predictions.")
    return combined


def get_prophet_forecast(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    horizon: int = 28,
) -> pd.DataFrame:
    """Generate 30-day forward forecasts for a list of series."""
    from joblib import Parallel, delayed

    logger.info(f"Generating {horizon}-day Prophet forecasts for {len(series_ids):,} series…")

    # Check if database is in-memory
    db_list = con.execute("PRAGMA database_list").df()
    db_file = db_list["file"].iloc[0]
    is_memory = db_file is None or db_file == ""

    if is_memory:
        results = [
            _prophet_worker(sid, None, horizon, is_forecast=True, con=con)
            for sid in tqdm(series_ids, desc="Prophet forecast")
        ]
    else:
        results = Parallel(n_jobs=4, backend="loky", verbose=0)(
            delayed(_prophet_worker)(sid, db_file, horizon, is_forecast=True)
            for sid in tqdm(series_ids, desc="Prophet forecast")
        )

    combined = pd.concat([r for r in results if r is not None], ignore_index=True)
    if not combined.empty:
        combined[["prophet_point", "prophet_lower", "prophet_upper"]] = (
            combined[["prophet_point", "prophet_lower", "prophet_upper"]].clip(lower=0)
        )
    return combined


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    sample = con.execute(
        "SELECT DISTINCT id FROM demand_classes WHERE demand_class='Smooth' LIMIT 5"
    ).df()["id"].tolist()
    preds = train_predict_prophet(con, sample, horizon=28)
    print(preds.tail(10))
    con.close()
