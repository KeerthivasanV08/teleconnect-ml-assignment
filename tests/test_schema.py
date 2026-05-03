"""
tests/test_schema.py

Lightweight schema validation tests for cleaned dataset.
"""

import sys
import os
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestDataSchema:
    """Validate cleaned dataset schema."""

    @pytest.fixture
    def df(self):
        """Load cleaned dataset."""
        data_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        return pd.read_csv(data_path)

    def test_dataset_exists(self):
        """Test that cleaned dataset exists."""
        data_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        assert data_path.exists(), "Cleaned dataset not found"

    def test_required_columns_exist(self, df):
        """Test that all required columns exist."""
        required_cols = ["churn", "monthlycharges"]
        for col in required_cols:
            assert col in df.columns, f"Required column '{col}' not found"

    def test_churn_column_exists(self, df):
        """Test churn column exists and has valid values."""
        assert "churn" in df.columns
        assert df["churn"].dtype == "object" or df["churn"].dtype == "int64"
        unique_values = df["churn"].unique()
        assert len(unique_values) == 2, "Churn column should have exactly 2 values"

    def test_monthlycharges_column_exists(self, df):
        """Test monthlycharges column exists and is numeric."""
        assert "monthlycharges" in df.columns
        assert pd.api.types.is_numeric_dtype(df["monthlycharges"])

    def test_no_empty_dataset(self, df):
        """Test dataset is not empty."""
        assert len(df) > 0, "Dataset is empty"
        assert df.shape[1] > 0, "Dataset has no columns"

    def test_dataset_shape(self, df):
        """Test dataset has expected rough shape."""
        # Should have at least 5000 rows, at least 15 columns
        assert df.shape[0] >= 5000, "Dataset has too few rows"
        assert df.shape[1] >= 15, "Dataset has too few columns"