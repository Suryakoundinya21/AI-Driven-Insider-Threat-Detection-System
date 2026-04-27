import numpy as np
import pandas as pd
import shap
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def build_shap_explainer(clf, X_background: np.ndarray):
    logger.info("Building SHAP TreeExplainer for Isolation Forest...")
    explainer = shap.TreeExplainer(clf)
    logger.info("SHAP explainer ready.")
    return explainer


def compute_shap_values(
    explainer,
    X: np.ndarray,
    feature_names: list,
    max_samples: int = 3000,
) -> pd.DataFrame:
    n = min(len(X), max_samples)
    logger.info(f"Computing SHAP values for {n:,} sessions...")
    X_sample  = X[:n]
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    logger.info(f"SHAP values computed. Shape: {shap_df.shape}")
    return shap_df


def plot_shap_summary(
    shap_df: pd.DataFrame,
    X: np.ndarray,
    feature_names: list,
    save_path: str,
    max_samples: int = 3000,
):
    n = min(len(X), max_samples)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_df.values,
        X[:n],
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    plt.title("SHAP Feature Importance — Isolation Forest")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved: {save_path}")


def get_top_shap_features(
    shap_df: pd.DataFrame,
    idx: int,
    top_n: int = 5,
) -> list:
    if idx >= len(shap_df):
        return []
    row = shap_df.iloc[idx]
    top = row.abs().nlargest(top_n)
    return [(feat, float(row[feat])) for feat in top.index]
