import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load model + data
# ------------------------------------------------------------
def load_model(model):
    return model


# ------------------------------------------------------------
# SHAP Explainer
# ------------------------------------------------------------
def get_shap_values(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    return explainer, shap_values


# ------------------------------------------------------------
# Global Importance Plot
# ------------------------------------------------------------
def plot_global_shap(shap_values):
    shap.summary_plot(shap_values, show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Local Explanation (single row)
# ------------------------------------------------------------
def plot_local_shap(explainer, X_sample):
    shap_values = explainer(X_sample)
    shap.plots.waterfall(shap_values[0])
    plt.show()


# ------------------------------------------------------------
# Partial Dependence (manual simple version)
# ------------------------------------------------------------
from sklearn.inspection import PartialDependenceDisplay

def plot_pdp(model, X, features):
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=features,
        grid_resolution=50
    )
    plt.show()