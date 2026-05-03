"""
tests/test_preprocessing.py

Lightweight preprocessing validation tests.
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_telco_data


class TestPreprocessing:
    """Validate preprocessing pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Load raw dataset for preprocessing test."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "Telco-Customer-Churn.csv"
        if data_path.exists():
            return pd.read_csv(data_path).head(100)  # Small sample
        else:
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
        """Test preprocessing removes duplicates (if any exist)."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        assert duplicates == 0, f"Dataset contains {duplicates} duplicate rows"

    def test_preprocessing_handles_missing_values(self):
        """Test preprocessing handles missing values."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)
        # After preprocessing, critical columns should not have nulls
        critical_cols = ["churn", "monthlycharges"]
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                assert null_count == 0, f"Column '{col}' has {null_count} null values after preprocessing"

    def test_preprocessing_output_not_all_nulls(self):
        """Test that preprocessing doesn't result in all-null columns."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df)
            assert null_pct < 1.0, f"Column '{col}' is 100% null after preprocessing"

    def test_target_variable_distribution(self):
        """Test target variable has reasonable distribution."""
        cleaned_path = Path(__file__).parent.parent / "data" / "processed" / "telco_cleaned.csv"
        df = pd.read_csv(cleaned_path)
        if "churn" in df.columns:
            value_counts = df["churn"].value_counts()
            # Both classes should be represented
            assert len(value_counts) == 2, "Churn should have exactly 2 classes"
            # Neither class should be extremely dominant (not >99%)
            for val, count in value_counts.items():
                pct = count / len(df)
                assert pct < 0.99, f"Class '{val}' is {pct*100:.1f}% - possibly imbalanced preprocessing"