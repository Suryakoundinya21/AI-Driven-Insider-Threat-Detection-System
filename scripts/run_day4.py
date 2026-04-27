import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging, json

from src.config import config
from src.modeling.preprocessor import prepare_data, split_normal_sessions
from src.modeling.autoencoder  import build_model
from src.modeling.trainer      import train_autoencoder
from src.modeling.scorer       import compute_reconstruction_errors, apply_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_training_history(history, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss", color="#378ADD")
    plt.plot(history["val_loss"],   label="Val Loss",   color="#D85A30")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder — Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_reconstruction_errors(df, threshold, save_path):
    errors  = df["reconstruction_error"].values
    flagged = df["ae_anomaly_flag"].values
    colors  = ["#D85A30" if f else "#378ADD" for f in flagged]

    plt.figure(figsize=(12, 4))
    plt.scatter(range(len(errors)), errors, c=colors, s=1, alpha=0.4)
    plt.axhline(y=threshold, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold={threshold:.4f}")
    plt.xlabel("Session index")
    plt.ylabel("Reconstruction error")
    plt.title("Autoencoder — Reconstruction Error per Session")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_score_distribution(df, save_path):
    normal  = df[df["ae_anomaly_flag"] == 0]["ae_anomaly_score"]
    anomaly = df[df["ae_anomaly_flag"] == 1]["ae_anomaly_score"]
    plt.figure(figsize=(10, 4))
    plt.hist(normal,  bins=80, alpha=0.6, color="#378ADD", label="Normal")
    plt.hist(anomaly, bins=80, alpha=0.6, color="#D85A30", label="Anomaly")
    plt.xlabel("Normalized anomaly score (0-1)")
    plt.ylabel("Count")
    plt.title("Anomaly Score Distribution — Normal vs Flagged")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    os.makedirs("reports/plots", exist_ok=True)

    # 1. Load
    logger.info("Loading feature matrix...")
    df = pd.read_parquet(config.FEATURES_DIR / "feature_matrix.parquet")
    logger.info(f"Loaded: {df.shape}")

    # 2. Split
    normal_df, all_df = split_normal_sessions(df)

    # 3. Scale
    X_train, features, scaler = prepare_data(normal_df, fit=True)
    X_all,   _,        _      = prepare_data(all_df, scaler=scaler, fit=False)
    input_dim = X_train.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    # 4. Build
    model = build_model(input_dim=input_dim, encoding_dim=8)

    # 5. Train
    model, history = train_autoencoder(
        model      = model,
        X_train    = X_train,
        epochs     = 50,
        batch_size = 1024,
        lr         = 1e-3,
        patience   = 8,
    )

    # 6. Plot training
    plot_training_history(history, "reports/plots/ae_training_history.png")

    # 7. Score
    logger.info("Scoring all sessions...")
    errors          = compute_reconstruction_errors(model, X_all)
    df_scored, thr  = apply_scores(all_df, errors)

    # 8. Plots
    plot_reconstruction_errors(df_scored, thr,
        "reports/plots/ae_reconstruction_errors.png")
    plot_score_distribution(df_scored,
        "reports/plots/ae_score_distribution.png")

    # 9. Summary
    print("\n" + "="*60)
    print("AUTOENCODER RESULTS SUMMARY")
    print("="*60)
    print(f"Total sessions scored : {len(df_scored):,}")
    print(f"Anomaly threshold     : {thr:.6f}")
    print(f"Anomalies detected    : {df_scored['ae_anomaly_flag'].sum():,} "
          f"({df_scored['ae_anomaly_flag'].mean()*100:.2f}%)")

    cols  = ["user", "date_only", "ae_anomaly_score",
             "reconstruction_error", "composite_risk_score",
             "device_count", "email_to_external", "http_suspicious"]
    avail = [c for c in cols if c in df_scored.columns]
    print("\nTop 10 anomalous sessions:")
    print(df_scored.nlargest(10, "ae_anomaly_score")[avail].to_string(index=False))

    # 10. Save
    out = config.FEATURES_DIR / "feature_matrix_ae_scored.parquet"
    df_scored.to_parquet(out, index=False)
    logger.info(f"Scored dataset saved: {out}")

    feat_path = config.MODELS_DIR / "autoencoder" / "features_used.json"
    with open(feat_path, "w") as f:
        json.dump(features, f)
    logger.info(f"Feature list saved: {feat_path}")

    print("\nPlots saved to reports/plots/")
    print("Day 4 Complete.")


if __name__ == "__main__":
    main()
