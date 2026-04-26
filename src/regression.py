"""
src/regression.py

Reusable utilities for Task 4 — Regression: Revenue Forecasting
Predict MonthlyCharges using remaining customer features.
"""

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


# ============================================================
# Data Preparation
# ============================================================
def prepare_regression_data(df):
    """
    Prepare dataset for regression modeling.

    Target:
        monthlycharges

    Drops:
        customerid, monthlycharges

    Steps:
        - split X and y
        - label encode categorical columns
        - train/test split
        - scale numeric features

    Returns:
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    """
    y = df["monthlycharges"]
    X = df.drop(columns=["customerid", "monthlycharges"])

    categorical_cols = X.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


# ============================================================
# Adjusted R²
# ============================================================
def adjusted_r2_score(r2, n, p):
    """
    Compute Adjusted R-squared.
    """
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


# ============================================================
# Evaluation Metrics
# ============================================================
def evaluate_regression(y_true, y_pred, n_features):
    """
    Compute regression metrics.

    Returns:
        dict of MAE, MSE, RMSE, R2, Adj_R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2_score(r2, len(y_true), n_features)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adj_R2": adj_r2
    }


# ============================================================
# Train + Tune Single Model
# ============================================================
def train_regressor(model, param_grid, X_train, y_train):
    """
    Tune and train one regression model using GridSearchCV.

    Returns:
        best_model, best_params, training_time
    """
    start = time.time()

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    end = time.time()

    return grid.best_estimator_, grid.best_params_, (end - start)


# ============================================================
# Run One Model End-to-End
# ============================================================
def run_regression_model(model_name, model, param_grid, X_train, X_test, y_train, y_test):
    """
    Full single-model workflow:
        - tune
        - train
        - predict
        - evaluate

    Returns:
        best_model, predictions, metrics dict
    """
    best_model, best_params, training_time = train_regressor(
        model, param_grid, X_train, y_train
    )

    y_pred = best_model.predict(X_test)

    metrics = evaluate_regression(y_test, y_pred, X_test.shape[1])
    metrics["Model"] = model_name
    metrics["Training_Time"] = training_time
    metrics["Best_Params"] = best_params

    return best_model, y_pred, metrics


# ============================================================
# Model Configurations
# ============================================================
def get_regression_models():
    """
    Returns all required regression models and parameter grids.
    """
    models = {
        "Linear Regression": (
            LinearRegression(),
            {"fit_intercept": [True, False]}
        ),

        "Ridge": (
            Ridge(),
            {"alpha": [0.01, 0.1, 1, 10]}
        ),

        "Lasso": (
            Lasso(max_iter=5000),
            {"alpha": [0.001, 0.01, 0.1, 1]}
        ),

        "ElasticNet": (
            ElasticNet(max_iter=5000),
            {
                "alpha": [0.001, 0.01, 0.1, 1],
                "l1_ratio": [0.2, 0.5, 0.8]
            }
        ),

        "Decision Tree": (
            DecisionTreeRegressor(random_state=42),
            {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        ),

        "Random Forest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None]
            }
        ),

        "SVR": (
            SVR(),
            {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        )
    }

    return models


# ============================================================
# Coefficient Analysis
# ============================================================
def get_coefficients(model, feature_names):
    """
    Return sorted coefficient importance for linear models.
    """
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    })

    coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)
    return coef_df


# ============================================================
# Residuals
# ============================================================
def get_residuals(y_true, y_pred):
    """
    Compute residuals.
    """
    return y_true - y_pred