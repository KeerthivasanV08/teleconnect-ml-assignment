"""
src/classification.py

Reusable utilities for Task 3 — Classification: Churn Prediction
Build and evaluate classifiers one-by-one.
"""

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")


# ============================================================
# Data Preparation
# ============================================================
def prepare_classification_data(df):
    """
    Prepare dataset for churn classification.

    Target:
        churn

    Drops:
        customerid

    Steps:
        - encode target
        - label encode categorical features
        - split into train / validation / test
        - scale numeric features

    Returns:
        X_train, X_val, X_test,
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    """
    df = df.copy()

    # Encode target
    df["churn"] = df["churn"].map({"No": 0, "Yes": 1}) if df["churn"].dtype == "object" else df["churn"]

    y = df["churn"]
    X = df.drop(columns=["customerid", "churn"])

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Validation / test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Scale
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    return (
        X_train, X_val, X_test,
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )


# ============================================================
# Train + Tune Single Classifier
# ============================================================
def train_classifier(model, param_grid, X_train, y_train):
    """
    Tune and train one classifier using GridSearchCV.
    """
    start = time.time()

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    end = time.time()

    return grid.best_estimator_, grid.best_params_, (end - start)


# ============================================================
# Evaluate Classifier
# ============================================================
def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate classification model.

    Returns:
        metrics dict
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        "Confusion_Matrix": confusion_matrix(y_test, y_pred),
        "Classification_Report": classification_report(y_test, y_pred, output_dict=True)
    }

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metrics["FPR"] = fpr
        metrics["TPR"] = tpr

    return y_pred, metrics


# ============================================================
# Run One Model End-to-End
# ============================================================
def run_classification_model(model_name, model, param_grid, X_train, X_test, y_train, y_test):
    """
    Full workflow for one classifier:
        - tune
        - train
        - predict
        - evaluate

    Returns:
        best_model, predictions, metrics dict
    """
    best_model, best_params, training_time = train_classifier(
        model, param_grid, X_train, y_train
    )

    y_pred, metrics = evaluate_classifier(best_model, X_test, y_test)

    metrics["Model"] = model_name
    metrics["Best_Params"] = best_params
    metrics["Training_Time"] = training_time

    return best_model, y_pred, metrics


# ============================================================
# Model Configurations
# ============================================================
def get_classification_models():
    """
    Returns all required classifiers and parameter grids.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        ),

        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None]
            }
        ),

        "SVM": (
            SVC(probability=True),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        ),

        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            }
        )
    }

    return models


# ============================================================
# Feature Importance
# ============================================================
def get_feature_importance(model, feature_names):
    """
    Return feature importance for tree-based models.
    """
    fi = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return fi