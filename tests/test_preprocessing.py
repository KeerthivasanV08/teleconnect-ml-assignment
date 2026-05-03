"""
tests/test_preprocessing.py

Lightweight preprocessing validation tests.
"""

import sys
import os
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_telco_data


class TestPreprocessing:
    """Validate preprocessing pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Load raw dataset for preprocessing test."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "Telco-Customer-Churn.csv"
        if data_path.exists():
            return pd.read_csv(data_path).head(100)
        pytest.skip("Raw data not available")

    def test_cleaned_data_exists(self):
        """Test cleaned data output exists."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        assert cleaned_path.exists(), "Cleaned data not found"

    def test_cleaned_data_not_empty(self):
        """Test cleaned data is not empty."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)
        assert len(df) > 0, "Cleaned dataset is empty"

    def test_preprocessing_removes_duplicates(self):
        """Test preprocessing removes duplicates."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)
        duplicates = df.duplicated().sum()
        assert duplicates == 0, f"Dataset contains {duplicates} duplicate rows"

    def test_preprocessing_handles_missing_values(self):
        """Test preprocessing handles missing values."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)

        critical_cols = ["churn", "monthlycharges"]
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                assert null_count == 0, f"Column '{col}' has {null_count} null values after preprocessing"

    def test_preprocessing_output_not_all_nulls(self):
        """Test preprocessing doesn't create fully null columns."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)

        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df)
            assert null_pct < 1.0, f"Column '{col}' is 100% null after preprocessing"

    def test_target_variable_distribution(self):
        """Test target variable has reasonable class distribution."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)

        if "churn" in df.columns:
            value_counts = df["churn"].value_counts()

            assert len(value_counts) == 2, "Churn should have exactly 2 classes"

            for val, count in value_counts.items():
                pct = count / len(df)
                assert pct < 0.99, f"Class '{val}' is {pct*100:.1f}% - possible imbalance issue"

    def test_preprocess_telco_data_pipeline(self):
        """Test full preprocessing pipeline function."""
        df = pd.DataFrame({
            "customerid": ["0001", "0002"],
            "gender": ["Male", "Female"],
            "tenure": [10, 20],
            "monthlycharges": [50.0, 80.0],
            "totalcharges": [500.0, 1600.0],
            "phoneservice": ["Yes", "No"],
            "multiplelines": ["No", "Yes"],
            "onlinesecurity": ["Yes", "No"],
            "onlinebackup": ["No", "Yes"],
            "deviceprotection": ["Yes", "No"],
            "techsupport": ["No", "Yes"],
            "streamingtv": ["Yes", "No"],
            "streamingmovies": ["No", "Yes"],
            "contract": ["Month-to-month", "Two year"],
            "churn": ["Yes", "No"]
        })

        processed_df, scaler = preprocess_telco_data(df)

        assert processed_df.shape[0] == 2
        assert "customerid" not in processed_df.columns
        assert "avgmonthlyspend" in processed_df.columns
        assert "servicecount" in processed_df.columns
        assert "remaining_contract_months" in processed_df.columns
        assert "contractvalue" in processed_df.columns
        assert "churn" in processed_df.columns
        assert scaler is not None