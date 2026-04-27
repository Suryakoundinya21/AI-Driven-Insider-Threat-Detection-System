import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

from src.config import config
from src.modeling.preprocessor    import prepare_data, split_normal_sessions
from src.modeling.isolation_forest import train_isolation_forest, score_isolation_forest
from src.modeling.ensemble        import build_ensemble_score
from src.modeling.evaluator       import (
    load_ground_truth, attach_ground_truth,
    evaluate_model, compare_models
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_score_comparison(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df["ae_anomaly_score"], bins=60,
                 color="#378ADD", alpha=0.7, label="AE Score")
    axes[0].set_title("Autoencoder Anomaly Score")
    axes[0].set_xlabel("Score (0-1)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["if_anomaly_score"], bins=60,
                 color="#1D9E75", alpha=0.7, label="IF Score")
    axes[1].set_title("Isolation Forest Anomaly Score")
    axes[1].set_xlabel("Score (0-1)")

    axes[2].hist(df["ensemble_score"], bins=60,
                 color="#D85A30", alpha=0.7, label="Ensemble")
    axes[2].set_title("Ensemble Score (AE + IF)")
    axes[2].set_xlabel("Score (0-1)")

    plt.suptitle("Anomaly Score Distributions — All Models", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_venn_overlap(df, save_path):
    ae_flags = set(df[df["ae_anomaly_flag"] == 1].index)
    if_flags = set(df[df["if_anomaly_flag"] == 1].index)

    ae_only   = len(ae_flags - if_flags)
    if_only   = len(if_flags - ae_flags)
    both      = len(ae_flags & if_flags)

    fig, ax = plt.subplots(figsize=(7, 5))
    from matplotlib.patches import Circle
    c1 = Circle((0.38, 0.5), 0.28, color="#378ADD", alpha=0.5, label=f"AE only: {ae_only}")
    c2 = Circle((0.62, 0.5), 0.28, color="#1D9E75", alpha=0.5, label=f"IF only: {if_only}")
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(0.28, 0.5, str(ae_only),  ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0.72, 0.5, str(if_only),  ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0.50, 0.5, str(both),     ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0.28, 0.82, "Autoencoder", ha="center", fontsize=11)
    ax.text(0.72, 0.82, "Isolation Forest", ha="center", fontsize=11)
    ax.text(0.50, 0.18, f"Both: {both}", ha="center", fontsize=10,
            color="#555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Model Agreement — Flagged Sessions Overlap", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_top_users(df, save_path):
    top = (
        df.groupby("user")["ensemble_score"]
        .max()
        .sort_values(ascending=False)
        .head(20)
    )
    colors = ["#D85A30" if s >= 0.8 else "#378ADD" for s in top.values]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(top)), top.values, color=colors)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Max ensemble score")
    ax.set_title("Top 20 Users by Maximum Ensemble Anomaly Score")
    ax.axhline(y=0.8, color="red", linestyle="--", linewidth=1,
               label="High-risk threshold (0.8)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_roc_curves(df, results, save_path):
    if not results or "is_insider" not in df.columns:
        logger.warning("Skipping ROC — no ground truth")
        return
    if df["is_insider"].sum() == 0:
        logger.warning("Skipping ROC — no insider sessions labeled")
        return

    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 6))

    configs = [
        ("ae_anomaly_score",  "#378ADD", "Autoencoder"),
        ("if_anomaly_score",  "#1D9E75", "Isolation Forest"),
        ("ensemble_score",    "#D85A30", "Ensemble"),
    ]
    for score_col, color, label in configs:
        if score_col not in df.columns:
            continue
        fpr, tpr, _ = roc_curve(df["is_insider"], df[score_col])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    os.makedirs("reports/plots", exist_ok=True)

    # 1. Load AE-scored feature matrix
    logger.info("Loading AE-scored feature matrix...")
    df = pd.read_parquet(
        config.FEATURES_DIR / "feature_matrix_ae_scored.parquet"
    )
    logger.info(f"Loaded: {df.shape}")

    # 2. Prepare features for IF (reuse AE scaler)
    normal_df, all_df = split_normal_sessions(df)
    X_train, features, scaler = prepare_data(normal_df, fit=False)
    X_all,   _,        _      = prepare_data(all_df,   scaler=scaler, fit=False)

    # 3. Train Isolation Forest
    clf = train_isolation_forest(
        X_train,
        contamination=config.CONTAMINATION,
        n_estimators=200,
    )

    # 4. Score all sessions with IF
    df_if = score_isolation_forest(clf, X_all, all_df)

    # 5. Merge AE scores into IF-scored df
    df_if["ae_anomaly_score"] = df["ae_anomaly_score"].values
    df_if["ae_anomaly_flag"]  = df["ae_anomaly_flag"].values
    df_if["reconstruction_error"] = df["reconstruction_error"].values

    # 6. Build ensemble score
    df_ensemble = build_ensemble_score(df_if, ae_weight=0.5, if_weight=0.5)

    # 7. Load and attach ground truth
    gt = load_ground_truth()
    df_labeled = attach_ground_truth(df_ensemble, gt)

    # 8. Evaluate all models
    results = []
    for flag_col, score_col, name in [
        ("ae_anomaly_flag",        "ae_anomaly_score", "Autoencoder"),
        ("if_anomaly_flag",        "if_anomaly_score", "Isolation Forest"),
        ("ensemble_flag_union",    "ensemble_score",   "Ensemble (Union)"),
        ("ensemble_flag_intersect","ensemble_score",   "Ensemble (Intersect)"),
        ("ensemble_flag_top5pct",  "ensemble_score",   "Ensemble (Top 5%)"),
    ]:
        r = evaluate_model(df_labeled, flag_col, score_col, name)
        if r:
            results.append(r)

    if results:
        comparison_df = compare_models(results)
        comparison_df.to_csv("reports/model_comparison.csv", index=False)
        logger.info("Model comparison saved: reports/model_comparison.csv")

    # 9. Plots
    plot_score_comparison(df_ensemble, "reports/plots/score_comparison.png")
    plot_venn_overlap(df_ensemble,     "reports/plots/venn_overlap.png")
    plot_top_users(df_ensemble,        "reports/plots/top_users_risk.png")
    plot_roc_curves(df_labeled, results,"reports/plots/roc_curves.png")

    # 10. Summary
    print("\n" + "="*60)
    print("DAY 5 SUMMARY")
    print("="*60)
    print(f"Total sessions       : {len(df_ensemble):,}")
    print(f"AE anomalies         : {df_ensemble['ae_anomaly_flag'].sum():,} "
          f"({df_ensemble['ae_anomaly_flag'].mean()*100:.2f}%)")
    print(f"IF anomalies         : {df_ensemble['if_anomaly_flag'].sum():,} "
          f"({df_ensemble['if_anomaly_flag'].mean()*100:.2f}%)")
    print(f"Ensemble union       : {df_ensemble['ensemble_flag_union'].sum():,} "
          f"({df_ensemble['ensemble_flag_union'].mean()*100:.2f}%)")
    print(f"Ensemble intersection: {df_ensemble['ensemble_flag_intersect'].sum():,} "
          f"({df_ensemble['ensemble_flag_intersect'].mean()*100:.2f}%)")

    print("\nTop 15 highest-risk sessions (ensemble):")
    cols = ["user", "date_only", "ensemble_score",
            "ae_anomaly_score", "if_anomaly_score",
            "device_count", "email_to_external",
            "http_suspicious", "composite_risk_score"]
    avail = [c for c in cols if c in df_ensemble.columns]
    print(df_ensemble.nlargest(15, "ensemble_score")[avail].to_string(index=False))

    # 11. Save final scored dataset
    out = config.FEATURES_DIR / "feature_matrix_ensemble_scored.parquet"
    df_ensemble.to_parquet(out, index=False)
    logger.info(f"Final ensemble dataset saved: {out}")
    print("\nDay 5 Complete.")


if __name__ == "__main__":
    main()
