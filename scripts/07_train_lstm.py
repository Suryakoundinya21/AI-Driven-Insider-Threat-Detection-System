import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging, json, joblib

from src.config import config
from src.modeling.lstm_sequence import (
    build_sequences, normalize_sequences, get_sequence_features
)
from src.modeling.lstm_model   import build_lstm_model
from src.modeling.lstm_trainer import train_lstm
from src.modeling.lstm_scorer  import (
    compute_lstm_errors, score_to_session
)
from src.modeling.evaluator import (
    load_ground_truth, attach_ground_truth,
    evaluate_model, compare_models
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_lstm_training(history, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss", color="#378ADD")
    plt.plot(history["val_loss"],   label="Val Loss",   color="#D85A30")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Autoencoder — Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_score_comparison(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color, label in [
        (axes[0], "ae_anomaly_score",  "#378ADD", "Autoencoder"),
        (axes[1], "if_anomaly_score",  "#1D9E75", "Isolation Forest"),
        (axes[2], "lstm_score",        "#D85A30", "LSTM"),
    ]:
        if col in df.columns:
            ax.hist(df[col], bins=60, color=color, alpha=0.7)
            ax.set_title(f"{label} Score Distribution")
            ax.set_xlabel("Score (0-1)")
            ax.set_ylabel("Count")
    plt.suptitle("Anomaly Score Distributions — All 3 Models", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    os.makedirs("reports/plots", exist_ok=True)
    os.makedirs("models/lstm",   exist_ok=True)

    # 1. Load ensemble-scored dataset
    logger.info("Loading ensemble-scored dataset...")
    df = pd.read_parquet(
        config.FEATURES_DIR / "feature_matrix_ensemble_scored.parquet"
    )
    logger.info(f"Loaded: {df.shape}")

    # 2. Build sequences
    logger.info("Building user sequences (window=7 days)...")
    X_all, meta, feats = build_sequences(df, window=7, step=1)
    logger.info(f"Total sequences: {len(X_all):,} | Shape: {X_all.shape}")

    # 3. Use only normal sessions for training
    normal_df = df[df["zscore_anomaly"] == 0].copy()
    X_normal, meta_normal, _ = build_sequences(normal_df, window=7, step=1)
    logger.info(f"Normal sequences for training: {len(X_normal):,}")

    # 4. Normalize
    X_normal_norm, mean, std = normalize_sequences(X_normal)
    X_all_norm, _, _         = normalize_sequences(X_all, mean, std)

    # Save normalization params
    np.save("models/lstm/seq_mean.npy", mean)
    np.save("models/lstm/seq_std.npy",  std)
    joblib.dump(feats, "models/lstm/seq_features.pkl")
    logger.info("Normalization params saved")

    # 5. Build and train LSTM
    input_dim = X_normal_norm.shape[2]
    model     = build_lstm_model(
        input_dim  = input_dim,
        hidden_dim = 64,
        latent_dim = 16,
        num_layers = 2,
    )

    model, history = train_lstm(
        model      = model,
        X_train    = X_normal_norm,
        epochs     = 50,
        batch_size = 512,
        lr         = 1e-3,
        patience   = 8,
    )

    # 6. Plot training
    plot_lstm_training(history, "reports/plots/lstm_training.png")

    # 7. Score all sequences
    logger.info("Scoring all sequences with LSTM...")
    errors  = compute_lstm_errors(model, X_all_norm)
    df_lstm, lstm_threshold = score_to_session(errors, meta, df)

    logger.info(f"LSTM scores merged. Shape: {df_lstm.shape}")

    # 8. Update ensemble with LSTM
    ae_w, if_w, lstm_w = 0.4, 0.3, 0.3
    df_lstm["ensemble_score_v2"] = (
        ae_w   * df_lstm["ae_anomaly_score"] +
        if_w   * df_lstm["if_anomaly_score"] +
        lstm_w * df_lstm["lstm_score"]
    )
    threshold_v2 = df_lstm["ensemble_score_v2"].quantile(0.95)
    df_lstm["ensemble_flag_v2"] = (
        df_lstm["ensemble_score_v2"] >= threshold_v2
    ).astype(int)

    all3_flag = (
        (df_lstm["ae_anomaly_flag"]        == 1) &
        (df_lstm["if_anomaly_flag"]        == 1) &
        (df_lstm["lstm_flag"]              == 1)
    ).astype(int)
    df_lstm["ensemble_flag_all3"] = all3_flag

    logger.info(f"Ensemble v2 (AE+IF+LSTM) top-5%  : {df_lstm['ensemble_flag_v2'].sum():,}")
    logger.info(f"All 3 models agree                : {all3_flag.sum():,}")

    # 9. Evaluate with ground truth
    gt        = load_ground_truth()
    df_labeled = attach_ground_truth(df_lstm, gt)

    results = []
    for flag_col, score_col, name in [
        ("ae_anomaly_flag",     "ae_anomaly_score",    "Autoencoder"),
        ("if_anomaly_flag",     "if_anomaly_score",    "Isolation Forest"),
        ("lstm_flag",           "lstm_score",           "LSTM"),
        ("ensemble_flag_intersect","ensemble_score",    "Ensemble AE+IF"),
        ("ensemble_flag_v2",    "ensemble_score_v2",   "Ensemble AE+IF+LSTM"),
        ("ensemble_flag_all3",  "ensemble_score_v2",   "All 3 Agree"),
    ]:
        if flag_col in df_labeled.columns and score_col in df_labeled.columns:
            r = evaluate_model(df_labeled, flag_col, score_col, name)
            if r:
                results.append(r)

    if results:
        comparison = compare_models(results)
        comparison.to_csv("reports/model_comparison_v2.csv", index=False)
        logger.info("Model comparison v2 saved: reports/model_comparison_v2.csv")

    # 10. Score distribution plot
    plot_score_comparison(df_lstm, "reports/plots/all3_score_distribution.png")

    # 11. Summary
    print("\n" + "="*65)
    print("LSTM MODEL RESULTS SUMMARY")
    print("="*65)
    print(f"Total sequences trained on : {len(X_normal_norm):,}")
    print(f"Total sequences scored     : {len(X_all_norm):,}")
    print(f"LSTM threshold             : {lstm_threshold:.6f}")
    print(f"LSTM anomalies             : {df_lstm['lstm_flag'].sum():,} "
          f"({df_lstm['lstm_flag'].mean()*100:.2f}%)")
    print(f"Ensemble v2 (AE+IF+LSTM)   : {df_lstm['ensemble_flag_v2'].sum():,}")
    print(f"All 3 models agree         : {all3_flag.sum():,}")

    print("\nTop 10 highest-risk sessions (LSTM score):")
    cols  = ["user","date_only","lstm_score","ae_anomaly_score",
             "if_anomaly_score","ensemble_score_v2",
             "device_count","email_to_external","http_suspicious"]
    avail = [c for c in cols if c in df_lstm.columns]
    print(df_lstm.nlargest(10, "lstm_score")[avail].to_string(index=False))

    # 12. Save final scored dataset
    out = config.FEATURES_DIR / "feature_matrix_lstm_scored.parquet"
    df_lstm.to_parquet(out, index=False)
    logger.info(f"LSTM-scored dataset saved: {out}")
    print("\nDay 11 Complete.")


if __name__ == "__main__":
    main()
