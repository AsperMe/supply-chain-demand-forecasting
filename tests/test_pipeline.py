"""Tests for the data pipeline."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestDemandClassification:
    """Test ADI × CV² demand classification logic."""

    def test_smooth_classification(self):
        from src.pipeline.classify import _compute_adi, _compute_cv2, _classify
        # Dense, regular sales → Smooth
        sales = pd.Series([5, 4, 6, 5, 5, 4, 6, 5, 5, 5] * 10)
        adi = _compute_adi(sales)
        cv2 = _compute_cv2(sales)
        assert adi < 1.32
        assert cv2 < 0.49
        assert _classify(adi, cv2) == "Smooth"

    def test_intermittent_classification(self):
        from src.pipeline.classify import _compute_adi, _compute_cv2, _classify
        # Sparse, consistent size → Intermittent
        sales = pd.Series([0, 0, 5, 0, 0, 5, 0, 0, 0, 5] * 10)
        adi = _compute_adi(sales)
        cv2 = _compute_cv2(sales)
        assert adi >= 1.32
        assert cv2 < 0.49
        assert _classify(adi, cv2) == "Intermittent"

    def test_lumpy_classification(self):
        from src.pipeline.classify import _compute_adi, _compute_cv2, _classify
        # Sparse AND variable → Lumpy
        sales = pd.Series([0, 0, 0, 15, 0, 0, 0, 0, 0, 1, 0, 0, 0, 30, 0] * 6)
        adi = _compute_adi(sales)
        cv2 = _compute_cv2(sales)
        assert adi >= 1.32
        assert cv2 >= 0.49
        assert _classify(adi, cv2) == "Lumpy"

    def test_zero_series(self):
        from src.pipeline.classify import _compute_adi, _compute_cv2, _classify
        sales = pd.Series([0] * 100)
        adi = _compute_adi(sales)
        assert adi == float("inf")

    def test_cv2_single_nonzero(self):
        from src.pipeline.classify import _compute_cv2
        sales = pd.Series([0, 0, 0, 5, 0, 0])
        cv2 = _compute_cv2(sales)
        assert cv2 == 0.0  # Only one non-zero, no variance


class TestFeatureEngineering:
    """Test feature engineering functions."""

    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range("2016-01-01", periods=100)
        return pd.DataFrame({
            "id": "test_item_CA_1",
            "date": dates,
            "sales": np.random.randint(0, 10, 100),
            "sell_price": np.random.uniform(1, 10, 100),
            "event_name_1": "",
            "event_type_1": "",
            "event_name_2": "",
            "event_type_2": "",
            "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
            "state_id": "CA",
        })

    def test_calendar_features(self, sample_df):
        from src.pipeline.features import _build_calendar_features
        result = _build_calendar_features(sample_df)
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "month_sin" in result.columns
        assert result["day_of_week"].between(0, 6).all()
        assert result["is_weekend"].isin([0, 1]).all()

    def test_lag_features(self, sample_df):
        from src.pipeline.features import _build_lag_features
        # Add id column properly
        result = _build_lag_features(sample_df)
        assert "sales_lag_7" in result.columns
        assert "sales_lag_28" in result.columns
        # First 28 rows should be NaN for lag_28
        assert result["sales_lag_28"].iloc[:28].isna().all()

    def test_rolling_features(self, sample_df):
        from src.pipeline.features import _build_rolling_features
        result = _build_rolling_features(sample_df)
        assert "rolling_mean_7" in result.columns
        assert "rolling_mean_28" in result.columns
        # Rolling means should be non-negative
        assert (result["rolling_mean_7"].dropna() >= 0).all()

    def test_event_features(self, sample_df):
        from src.pipeline.features import _build_event_features
        result = _build_event_features(sample_df)
        assert "is_sporting" in result.columns
        assert "snap_active" in result.columns
        assert result["is_sporting"].isin([0, 1]).all()


class TestEvaluationMetrics:
    """Test RMSE, MAE, SMAPE calculations."""

    def test_rmse_perfect(self):
        from src.ensemble.evaluate import rmse
        a = np.array([1, 2, 3, 4, 5])
        assert rmse(a, a) == 0.0

    def test_mae_perfect(self):
        from src.ensemble.evaluate import mae
        a = np.array([1, 2, 3, 4, 5])
        assert mae(a, a) == 0.0

    def test_rmse_known_value(self):
        from src.ensemble.evaluate import rmse
        actual = np.array([3.0, 3.0, 3.0, 3.0])
        pred   = np.array([5.0, 5.0, 5.0, 5.0])
        assert abs(rmse(actual, pred) - 2.0) < 1e-6

    def test_smape_zero_actual(self):
        from src.ensemble.evaluate import smape
        actual = np.array([0.0, 0.0])
        pred   = np.array([0.0, 0.0])
        # When both are zero, smape should return 0 (not nan)
        result = smape(actual, pred)
        assert result == 0.0

    def test_metrics_non_negative(self):
        from src.ensemble.evaluate import rmse, mae, smape
        actual = np.random.uniform(0, 10, 100)
        pred   = np.random.uniform(0, 10, 100)
        assert rmse(actual, pred) >= 0
        assert mae(actual, pred) >= 0
        assert smape(actual, pred) >= 0
