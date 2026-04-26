from pathlib import Path
import warnings
import pandas as pd
import numpy as np


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV safely.

    Parameters
    ----------
    filepath : str
        Path to raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found at: {filepath}")

    df = pd.read_csv(path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate dataset schema against expected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    actual_columns = df.columns.tolist()

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in actual_columns]
    unexpected_cols = [col for col in actual_columns if col not in EXPECTED_COLUMNS]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if unexpected_cols:
        warnings.warn(f"Unexpected columns found: {unexpected_cols}", UserWarning)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names:
    - strip spaces
    - lowercase
    - replace spaces with underscores

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized column names.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Inspect dataframe structure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    dict
        Summary dictionary with shape, dtypes, null counts, duplicates.
    """
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "null_counts": df.isnull().sum(),
        "duplicate_rows": df.duplicated().sum(),
    }
    return summary


def fix_totalcharges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix TotalCharges column:
    - convert to numeric
    - coerce invalid values to NaN

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Updated dataframe.
    """
    df = df.copy()

    if "totalcharges" not in df.columns:
        raise KeyError("'totalcharges' column not found. Run standardize_column_names() first.")

    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - numeric columns -> median
    - categorical columns -> mode

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed missing values.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def treat_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Treat outliers using IQR capping (winsorization).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list
        List of numeric columns for outlier treatment.

    Returns
    -------
    pd.DataFrame
        Dataframe with capped outliers.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def save_clean_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save cleaned dataframe to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    filepath : str
        Output path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)