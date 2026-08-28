"""Tests for the demand forecasting models."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def mock_db():
    """Create an in-memory DuckDB database with mock data for testing models."""
    con = duckdb.connect(":memory:")
    
    # 1. Create calendar
    dates = pd.date_range("2016-01-01", periods=100)
    cal_df = pd.DataFrame({
        "date": dates,
        "d": [f"d_{i+1}" for i in range(100)],
        "wm_yr_wk": [11601 + (i // 7) for i in range(100)],
        "weekday": dates.strftime("%A"),
        "wday": [i % 7 + 1 for i in range(100)],
        "month": dates.month,
        "year": dates.year,
        "event_name_1": ["SuperBowl" if i == 37 else "" for i in range(100)],
        "event_type_1": ["Sporting" if i == 37 else "" for i in range(100)],
        "event_name_2": "",
        "event_type_2": "",
        "snap_CA": [1 if i % 10 < 3 else 0 for i in range(100)],
        "snap_TX": [1 if i % 10 >= 3 and i % 10 < 6 else 0 for i in range(100)],
        "snap_WI": [1 if i % 10 >= 6 and i % 10 < 9 else 0 for i in range(100)],
    })
    con.execute("CREATE TABLE raw_calendar AS SELECT * FROM cal_df")
    
    # 2. Create raw sales
    items = ["HOBBIES_1_001_CA_1", "HOBBIES_1_002_CA_1"]
    sales_data = []
    for item in items:
        for i, dt in enumerate(dates):
            # Smooth/dense for first, intermittent/sparse for second
            val = np.random.randint(1, 10) if item == items[0] else (5 if i % 5 == 0 else 0)
            sales_data.append({
                "id": f"{item}_evaluation",
                "item_id": item,
                "dept_id": "HOBBIES_1",
                "cat_id": "HOBBIES",
                "store_id": "CA_1",
                "state_id": "CA",
                "d": f"d_{i+1}",
                "sales": val
            })
    sales_df = pd.DataFrame(sales_data)
    con.execute("CREATE TABLE raw_sales AS SELECT * FROM sales_df")
    
    # 3. Create sell prices
    prices_data = []
    for item in items:
        for wk in cal_df["wm_yr_wk"].unique():
            prices_data.append({
                "store_id": "CA_1",
                "item_id": item,
                "wm_yr_wk": wk,
                "sell_price": 5.99
            })
    prices_df = pd.DataFrame(prices_data)
    con.execute("CREATE TABLE raw_prices AS SELECT * FROM prices_df")
    
    # 4. Build master table
    from src.pipeline.ingest import _build_master_table
    _build_master_table(con)
    
    # 5. Build features table
    from src.pipeline.features import build_features
    build_features(con, chunk_size=10)
    
    # 6. Classify demand
    from src.pipeline.classify import classify_demand
    classify_demand(con)
    
    yield con
    con.close()


def test_lgbm_model(mock_db):
    """Test LightGBM training and prediction flow."""
    from src.models.lgbm_model import train_lgbm, predict_lgbm, get_feature_importance
    
    series_ids = ["HOBBIES_1_001_CA_1_evaluation", "HOBBIES_1_002_CA_1_evaluation"]
    
    # We specify a small validation window since our series is only 100 days
    models = train_lgbm(mock_db, series_ids=series_ids, val_days=10)
    assert "point" in models
    assert "q10" in models
    assert "q90" in models
    
    preds = predict_lgbm(mock_db, models, series_ids=series_ids)
    assert not preds.empty
    assert "lgbm_point" in preds.columns
    assert "lgbm_q10" in preds.columns
    assert "lgbm_q90" in preds.columns
    assert len(preds) > 0
    
    importance = get_feature_importance(models["point"])
    assert not importance.empty
    assert "feature" in importance.columns
    assert "importance" in importance.columns


def test_sarima_model(mock_db):
    """Test SARIMA fitting and forecasting."""
    from src.models.sarima_model import train_predict_sarima, get_sarima_forecast
    
    series_ids = ["HOBBIES_1_001_CA_1_evaluation"]
    
    preds = train_predict_sarima(mock_db, series_ids=series_ids, horizon=5)
    assert not preds.empty
    assert "sarima_point" in preds.columns
    
    fc = get_sarima_forecast(mock_db, series_ids=series_ids, horizon=5)
    assert not fc.empty
    assert "sarima_point" in fc.columns
    assert "sarima_lower" in fc.columns
    assert "sarima_upper" in fc.columns


def test_prophet_model(mock_db):
    """Test Prophet fitting and forecasting."""
    from src.models.prophet_model import train_predict_prophet, get_prophet_forecast
    
    series_ids = ["HOBBIES_1_001_CA_1_evaluation"]
    
    preds = train_predict_prophet(mock_db, series_ids=series_ids, horizon=5)
    assert not preds.empty
    assert "prophet_point" in preds.columns
    
    fc = get_prophet_forecast(mock_db, series_ids=series_ids, horizon=5)
    assert not fc.empty
    assert "prophet_point" in fc.columns
    assert "prophet_lower" in fc.columns
    assert "prophet_upper" in fc.columns


def test_dual_phase_model(mock_db):
    """Test Dual-Phase occurrence/magnitude model for sparse series."""
    from src.models.dual_phase import train_dual_phase, predict_dual_phase
    
    # Use the intermittent series
    series_ids = ["HOBBIES_1_002_CA_1_evaluation"]
    
    clf, reg = train_dual_phase(mock_db, series_ids=series_ids, val_days=10)
    assert clf is not None
    assert reg is not None
    
    preds = predict_dual_phase(mock_db, series_ids=series_ids, clf=clf, reg=reg)
    assert not preds.empty
    assert "dual_point" in preds.columns
    assert "dual_occurrence_prob" in preds.columns
