from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# ============================================================
# Load Clean Data
# ============================================================
def load_clean_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


# ============================================================
# Drop Non-Predictive Columns
# ============================================================
def drop_identifier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df.drop(columns=["customerid"], errors="ignore")


# ============================================================
# Encode Targets
# ============================================================
def encode_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["churn"] = df["churn"].map({"No": 0, "Yes": 1})
    return df


# ============================================================
# Create Derived Features
# ============================================================
def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avoid division by zero
    df["avgmonthlyspend"] = df["totalcharges"] / df["tenure"].replace(0, 1)

    service_cols = [
        "phoneservice", "multiplelines", "onlinesecurity", "onlinebackup",
        "deviceprotection", "techsupport", "streamingtv", "streamingmovies"
    ]

    df["servicecount"] = df[service_cols].apply(
        lambda row: sum(val in ["Yes", "DSL", "Fiber optic"] for val in row), axis=1
    )

    contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    df["remaining_contract_months"] = df["contract"].map(contract_map)
    df["contractvalue"] = df["monthlycharges"] * df["remaining_contract_months"]

    return df


# ============================================================
# Label Encoding (Binary Columns)
# ============================================================
def label_encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == "object"]

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    return df


# ============================================================
# One-Hot Encoding (Multi-Class Categorical)
# ============================================================
def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


# ============================================================
# Feature Scaling
# ============================================================
def scale_features(df: pd.DataFrame, method="standard"):
    df = df.copy()

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "churn"]

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler


# ============================================================
# Feature Selection Methods
# ============================================================
def correlation_filter(df: pd.DataFrame, threshold=0.85):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return drop_cols


def rfe_selection(X, y, n_features=15):
    model = LogisticRegression(max_iter=1000)
    selector = RFE(model, n_features_to_select=n_features)
    selector.fit(X, y)
    selected = X.columns[selector.support_].tolist()
    return selected


def tree_feature_importance(X, y, top_n=15):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False).head(top_n)


def mutual_info_selection(X, y, top_n=15):
    mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns)
    return mi_scores.sort_values(ascending=False).head(top_n)


# ============================================================
# Handle Class Imbalance
# ============================================================
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


def apply_random_undersampling(X, y):
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X, y)


def get_class_weights(y):
    counts = y.value_counts()
    total = len(y)
    return {cls: total / (len(counts) * count) for cls, count in counts.items()}


# ============================================================
# Train / Validation / Test Split
# ============================================================
def split_data(X, y, stratify=True):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y if stratify else None
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp if stratify else None
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Save Artifacts
# ============================================================
def save_scaler(scaler, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)