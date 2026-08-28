from __future__ import annotations

import pickle
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"
MODEL_DIR = ROOT / "data" / "models"

FEATURE_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "day_of_week", "week_of_year", "month", "year", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_sporting", "is_cultural", "is_national", "is_religious",
    "has_event", "snap_active",
    "sell_price", "price_change_pct", "is_promotion", "days_since_price_change",
    "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_lag_56",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
    "rolling_min_7", "rolling_max_7",
    "days_since_last_sale", "rolling_nonzero_rate_28",
]
CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

CLASSIFIER_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 10,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}

REGRESSOR_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 5,    # fewer non-zero rows available
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}


def _load_and_prepare(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    val_days: int = 28,
) -> dict:
    """Load features and prepare train/val splits for both stages."""
    ids_str = ", ".join(f"'{i}'" for i in series_ids)
    df = con.execute(
        f"SELECT * FROM features WHERE id IN ({ids_str}) ORDER BY id, date"
    ).df().dropna(subset=["sales_lag_56"])

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    y_occ = (df["sales"] > 0).astype(int)     
    y_mag = df["sales"].astype(float)           

    # Time-based split
    dates = pd.to_datetime(df["date"])
    cutoff = dates.max() - pd.Timedelta(days=val_days)
    train_mask = dates <= cutoff
    val_mask = dates > cutoff

    # Magnitude model uses ONLY non-zero rows in training
    nonzero_train = train_mask & (df["sales"] > 0)

    return {
        "X": X, "df": df, "dates": dates,
        "train_mask": train_mask, "val_mask": val_mask,
        "nonzero_train": nonzero_train,
        "y_occ": y_occ, "y_mag": y_mag,
        "available": available,
    }


def train_dual_phase(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    val_days: int = 28,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]:
    """
    Train the dual-phase model for intermittent/lumpy demand.

    Returns:
        (occurrence_classifier, magnitude_regressor)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training Dual-Phase model on {len(series_ids):,} intermittent/lumpy series…")

    data = _load_and_prepare(con, series_ids, val_days)
    X, y_occ, y_mag = data["X"], data["y_occ"], data["y_mag"]
    tm, vm, nz = data["train_mask"], data["val_mask"], data["nonzero_train"]

    logger.info(f"  Train rows: {tm.sum():,} | Non-zero train rows: {nz.sum():,} | Val rows: {vm.sum():,}")

    # Stage 1: Occurrence Classifier
    logger.info("  Training Stage 1: Occurrence Classifier…")
    clf = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
    clf.fit(
        X[tm], y_occ[tm],
        eval_set=[(X[vm], y_occ[vm])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
    )
    val_occ_preds = clf.predict_proba(X[vm])[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_occ[vm], val_occ_preds)
    logger.success(f"  Occurrence classifier: AUC={auc:.4f}, best_iter={clf.best_iteration_}")

    # Stage 2: Magnitude Regressor
    logger.info("  Training Stage 2: Magnitude Regressor (non-zero demand rows only)…")
    reg = lgb.LGBMRegressor(**REGRESSOR_PARAMS)

    # Validation: non-zero rows in val set
    nonzero_val = vm & (y_mag > 0)
    reg.fit(
        X[nz], y_mag[nz],
        eval_set=[(X[nonzero_val], y_mag[nonzero_val])] if nonzero_val.sum() > 0 else None,
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
    )
    logger.success(f"  Magnitude regressor: best_iter={reg.best_iteration_}")

    # Save
    with open(MODEL_DIR / "dual_phase_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODEL_DIR / "dual_phase_regressor.pkl", "wb") as f:
        pickle.dump(reg, f)
    logger.info("  Dual-phase models saved.")

    return clf, reg


def predict_dual_phase(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    clf: lgb.LGBMClassifier | None = None,
    reg: lgb.LGBMRegressor | None = None,
) -> pd.DataFrame:
    """
    Generate dual-phase predictions.

    Returns DataFrame: id, date, dual_occurrence_prob, dual_magnitude, dual_point
    Where:
      dual_occurrence_prob = P(demand > 0)
      dual_magnitude       = E(demand | demand > 0)
      dual_point           = dual_occurrence_prob × dual_magnitude
    """
    if clf is None:
        with open(MODEL_DIR / "dual_phase_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
    if reg is None:
        with open(MODEL_DIR / "dual_phase_regressor.pkl", "rb") as f:
            reg = pickle.load(f)

    ids_str = ", ".join(f"'{i}'" for i in series_ids)
    df = con.execute(
        f"SELECT * FROM features WHERE id IN ({ids_str}) ORDER BY id, date"
    ).df().dropna(subset=["sales_lag_56"])

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    occ_prob = clf.predict_proba(X)[:, 1]
    magnitude = np.clip(reg.predict(X), 0, None)
    combined = occ_prob * magnitude

    preds = df[["id", "date"]].copy()
    preds["dual_occurrence_prob"] = occ_prob.round(4)
    preds["dual_magnitude"] = magnitude.round(3)
    preds["dual_point"] = combined.round(3)

    logger.success(f"Dual-phase predictions: {len(preds):,} rows.")
    return preds


def get_dual_phase_forecast(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    horizon: int = 28,
    clf=None, reg=None,
) -> pd.DataFrame:
    """
    Generate forward forecasts using the latest available features.
    Note: For true forward forecast, features must be pre-computed with
    future calendar data (dates with no actuals).
    """
    preds = predict_dual_phase(con, series_ids, clf, reg)
    # Return last `horizon` rows per series (proxy forward forecast)
    return (
        preds.sort_values(["id", "date"])
        .groupby("id")
        .tail(horizon)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    sample = con.execute(
        "SELECT DISTINCT id FROM demand_classes WHERE demand_class IN ('Intermittent','Lumpy') LIMIT 200"
    ).df()["id"].tolist()
    if sample:
        clf, reg = train_dual_phase(con, sample)
        preds = predict_dual_phase(con, sample[:10], clf, reg)
        print(preds.describe())
    else:
        print("No intermittent/lumpy series found. Run classify.py first.")
    con.close()
