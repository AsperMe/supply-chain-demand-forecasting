from __future__ import annotations

import pickle
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"
MODEL_DIR = ROOT / "data" / "models"

# Feature columns used for training
FEATURE_COLS = [
    # IDs (as categoricals)
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    # Calendar
    "day_of_week", "week_of_year", "month", "year", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Events
    "is_sporting", "is_cultural", "is_national", "is_religious",
    "has_event", "snap_active",
    # Price
    "sell_price", "price_change_pct", "is_promotion", "days_since_price_change",
    # Lags
    "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_lag_56",
    # Rolling
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
    "rolling_min_7", "rolling_max_7",
    # Demand
    "days_since_last_sale", "rolling_nonzero_rate_28",
]

CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

# LightGBM base hyperparameters
BASE_PARAMS = {
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 20,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}


def _load_features(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load features table, optionally filtered to given series."""
    cols = ["id", "date", "sales"] + FEATURE_COLS
    cols_sql = ", ".join(dict.fromkeys(cols))
    if series_ids:
        ids_str = ", ".join(f"'{i}'" for i in series_ids)
        q = f"SELECT {cols_sql} FROM features WHERE id IN ({ids_str}) ORDER BY id, date"
    else:
        q = f"SELECT {cols_sql} FROM features ORDER BY id, date"
    df = con.execute(q).df()
    # Drop rows where lag features are all null
    df = df.dropna(subset=["sales_lag_56"])
    return df


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into X and y, encode categoricals."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df["sales"].astype("float32")
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X, y


def train_lgbm(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str] | None = None,
    quantiles: list[float] | None = None,
    val_days: int = 28,
) -> dict[str, lgb.LGBMRegressor]:
    """
    Train LightGBM models (point + quantiles).

    Args:
        con: DuckDB connection.
        series_ids: IDs to train on (None = all).
        quantiles: Quantile levels for probabilistic forecasting (e.g. [0.1, 0.9]).
        val_days: Number of days to hold out for validation.

    Returns:
        Dict of {objective_name: trained_model}
    """
    quantiles = quantiles or [0.1, 0.5, 0.9]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features for LightGBM…")
    df = _load_features(con, series_ids)
    X, y = _prepare(df)

    # Time-based train/val split
    dates = pd.to_datetime(df["date"])
    cutoff = dates.max() - pd.Timedelta(days=val_days)
    train_mask = dates <= cutoff
    val_mask = dates > cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    logger.info(f"  Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    models = {}

    # Point forecast (MAE objective)
    logger.info("Training LightGBM point forecast model…")
    point_params = {**BASE_PARAMS, "objective": "regression_l1", "metric": "mae"}
    point_model = lgb.LGBMRegressor(**point_params)
    point_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    models["point"] = point_model
    logger.success(f"  Point model: best iter={point_model.best_iteration_}")

    # Quantile forecasts
    for q in quantiles:
        logger.info(f"Training LightGBM quantile model (α={q})…")
        q_params = {
            **BASE_PARAMS,
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
        }
        q_model = lgb.LGBMRegressor(**q_params)
        q_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
        models[f"q{int(q*100)}"] = q_model
        logger.success(f"  Quantile α={q}: best iter={q_model.best_iteration_}")

    # Save models
    for name, model in models.items():
        path = MODEL_DIR / f"lgbm_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  Saved: {path}")

    return models


def predict_lgbm(
    con: duckdb.DuckDBPyConnection,
    models: dict[str, lgb.LGBMRegressor] | None = None,
    series_ids: list[str] | None = None,
    horizon: int = 28,
) -> pd.DataFrame:
    """
    Generate predictions from trained LightGBM models.

    Returns DataFrame with columns:
      id, date, lgbm_point, lgbm_q10, lgbm_q50, lgbm_q90
    """
    if models is None:
        models = {}
        for name in ["point", "q10", "q50", "q90"]:
            path = MODEL_DIR / f"lgbm_{name}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)

    df = _load_features(con, series_ids)
    X, _ = _prepare(df)

    preds = df[["id", "date"]].copy()
    for name, model in models.items():
        raw = model.predict(X)
        preds[f"lgbm_{name}"] = np.clip(raw, 0, None).round(3)

    return preds


def get_feature_importance(
    model: lgb.LGBMRegressor,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top N features by importance."""
    available = [c for c in FEATURE_COLS if c in model.feature_name_]
    imp = pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp.head(top_n)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    # Quick smoke test on a small subset
    sample = con.execute(
        "SELECT DISTINCT id FROM demand_classes WHERE demand_class='Smooth' LIMIT 100"
    ).df()["id"].tolist()
    models = train_lgbm(con, series_ids=sample)
    preds = predict_lgbm(con, models, series_ids=sample)
    print(preds.head())
    print(get_feature_importance(models["point"]))
    con.close()
