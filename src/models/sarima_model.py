from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"

# Fixed SARIMA order (pre-tuned for M5 daily sales, weekly seasonality)
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL = (1, 0, 1, 7)   # Weekly seasonality


def _get_series(
    con: duckdb.DuckDBPyConnection,
    series_id: str,
) -> pd.Series:
    """Return daily sales for one series as a pandas Series with date index."""
    df = con.execute(
        "SELECT date, sales FROM master WHERE id = ? ORDER BY date",
        [series_id],
    ).df()
    s = pd.Series(
        df["sales"].values,
        index=pd.to_datetime(df["date"]),
        name=series_id,
    )
    return s


def _fit_sarima(
    series: pd.Series,
    order: tuple = DEFAULT_ORDER,
    seasonal_order: tuple = DEFAULT_SEASONAL,
    horizon: int = 28,
) -> dict | None:
    """Fit SARIMAX and return predictions dict."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    try:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(disp=False, maxiter=200)

        # In-sample
        in_sample = fit.get_prediction()
        in_df = pd.DataFrame({
            "date": series.index,
            "sarima_point": np.clip(in_sample.predicted_mean.values, 0, None),
        })

        # Out-of-sample forecast
        forecast = fit.get_forecast(steps=horizon)
        fc_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
        fc_df = pd.DataFrame({
            "date": fc_dates,
            "sarima_point": np.clip(forecast.predicted_mean.values, 0, None),
            "sarima_lower": np.clip(forecast.conf_int().iloc[:, 0].values, 0, None),
            "sarima_upper": np.clip(forecast.conf_int().iloc[:, 1].values, 0, None),
        })

        return {"id": series.name, "in_sample": in_df, "forecast": fc_df}
    except Exception as e:
        logger.warning(f"  SARIMA failed for {series.name}: {e}")
        return None


def auto_select_order(
    con: duckdb.DuckDBPyConnection,
    sample_size: int = 20,
    p_range: range = range(0, 3),
    q_range: range = range(0, 3),
) -> tuple:
    """
    Grid search SARIMA orders on a random sample of Smooth series.
    Returns best (order, seasonal_order) tuple by average AIC.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    logger.info(f"Auto-selecting SARIMA order on {sample_size} sample series…")
    sample_ids = con.execute(
        f"SELECT id FROM demand_classes WHERE demand_class='Smooth' "
        f"ORDER BY RANDOM() LIMIT {sample_size}"
    ).df()["id"].tolist()

    best_aic = float("inf")
    best_order = DEFAULT_ORDER
    best_seasonal = DEFAULT_SEASONAL

    for p, q in product(p_range, q_range):
        order = (p, 1, q)
        seasonal = (1, 0, 1, 7)
        aics = []
        for sid in sample_ids[:5]:  # quick check on 5 series
            try:
                s = _get_series(con, sid)
                m = SARIMAX(s, order=order, seasonal_order=seasonal,
                            enforce_stationarity=False, enforce_invertibility=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = m.fit(disp=False, maxiter=100)
                aics.append(fit.aic)
            except Exception:
                pass
        if aics:
            avg_aic = np.mean(aics)
            logger.debug(f"  SARIMA{order}x{seasonal} AIC={avg_aic:.1f}")
            if avg_aic < best_aic:
                best_aic = avg_aic
                best_order = order
                best_seasonal = seasonal

    logger.success(f"Best SARIMA order: {best_order}x{best_seasonal} (AIC={best_aic:.1f})")
    return best_order, best_seasonal


def train_predict_sarima(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    horizon: int = 28,
    order: tuple | None = None,
    seasonal_order: tuple | None = None,
) -> pd.DataFrame:
    """
    Fit SARIMA on each series and return combined in-sample predictions.

    Returns: DataFrame with id, date, sarima_point
    """
    order = order or DEFAULT_ORDER
    seasonal_order = seasonal_order or DEFAULT_SEASONAL

    logger.info(f"Training SARIMA on {len(series_ids):,} series…")
    all_preds = []

    for sid in tqdm(series_ids, desc="SARIMA"):
        series = _get_series(con, sid)
        if len(series) < 60:
            continue
        result = _fit_sarima(series, order, seasonal_order, horizon)
        if result:
            df = result["in_sample"].copy()
            df["id"] = sid
            all_preds.append(df)

    combined = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    logger.success(f"SARIMA: {len(combined):,} prediction rows.")
    return combined


def get_sarima_forecast(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    horizon: int = 28,
) -> pd.DataFrame:
    """Generate forward forecasts from SARIMA."""
    all_fc = []
    for sid in tqdm(series_ids, desc="SARIMA forecast"):
        series = _get_series(con, sid)
        if len(series) < 60:
            continue
        result = _fit_sarima(series, horizon=horizon)
        if result:
            df = result["forecast"].copy()
            df["id"] = sid
            all_fc.append(df)
    return pd.concat(all_fc, ignore_index=True) if all_fc else pd.DataFrame()


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    sample = con.execute(
        "SELECT DISTINCT id FROM demand_classes WHERE demand_class='Smooth' LIMIT 3"
    ).df()["id"].tolist()
    preds = train_predict_sarima(con, sample)
    print(preds.tail(10))
    con.close()
