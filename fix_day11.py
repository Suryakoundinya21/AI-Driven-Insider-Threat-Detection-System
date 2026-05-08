import os

files = {}

# ─────────────────────────────────────────────
# LSTM SEQUENCE BUILDER
# ─────────────────────────────────────────────
files["src/modeling/lstm_sequence.py"] = """\
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

SEQUENCE_FEATURES = [
    "logon_count", "device_count", "email_count",
    "file_count", "http_count",
    "logon_after_hours", "device_after_hours",
    "email_after_hours", "file_after_hours", "http_after_hours",
    "first_logon_hour", "session_span_hours",
    "email_to_external", "http_suspicious",
    "activity_entropy", "total_after_hours",
    "after_hours_ratio", "composite_risk_score",
]


def get_sequence_features(df: pd.DataFrame) -> list:
    return [f for f in SEQUENCE_FEATURES if f in df.columns]


def build_sequences(
    df: pd.DataFrame,
    window: int = 7,
    step: int   = 1,
) -> tuple:
    feats   = get_sequence_features(df)
    df      = df.sort_values(["user", "date_only"]).copy()

    X_seqs  = []
    meta    = []

    logger.info(f"Building sequences: window={window}, features={len(feats)}")

    for user, user_df in df.groupby("user"):
        user_df = user_df.reset_index(drop=True)
        n       = len(user_df)

        if n < window + 1:
            continue

        vals = user_df[feats].fillna(0).values.astype(np.float32)

        for i in range(0, n - window, step):
            seq = vals[i : i + window]
            X_seqs.append(seq)
            meta.append({
                "user"      : user,
                "date_only" : str(user_df.loc[i + window - 1, "date_only"])[:10],
                "seq_start" : str(user_df.loc[i, "date_only"])[:10],
                "seq_idx"   : i,
            })

    X = np.array(X_seqs, dtype=np.float32)
    logger.info(f"Sequences built: {X.shape} — {len(meta)} total")
    return X, meta, feats


def normalize_sequences(
    X: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray  = None,
) -> tuple:
    if mean is None:
        mean = X.mean(axis=(0, 1), keepdims=True)
        std  = X.std(axis=(0, 1),  keepdims=True)
        std  = np.where(std == 0, 1e-6, std)
    X_norm = (X - mean) / std
    return X_norm, mean, std
"""

# ─────────────────────────────────────────────
# LSTM MODEL
# ─────────────────────────────────────────────
files["src/modeling/lstm_model.py"] = """\
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim   : int,
        hidden_dim  : int = 64,
        latent_dim  : int = 16,
        num_layers  : int = 2,
        dropout     : float = 0.2,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder — Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0,
            bidirectional = True,
        )

        # Bottleneck
        self.fc_enc = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)

        # Decoder — Unidirectional LSTM
        self.decoder = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.relu         = nn.ReLU()

    def forward(self, x):
        B, T, F = x.shape

        # Encode
        enc_out, _ = self.encoder(x)
        # Use last timestep from both directions
        enc_last   = enc_out[:, -1, :]
        latent     = self.relu(self.fc_enc(enc_last))

        # Decode — repeat latent across timesteps
        dec_input  = self.relu(self.fc_dec(latent))
        dec_input  = dec_input.unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_input)
        recon      = self.output_layer(dec_out)

        return recon

    def encode(self, x):
        enc_out, _ = self.encoder(x)
        enc_last   = enc_out[:, -1, :]
        return self.relu(self.fc_enc(enc_last))


def build_lstm_model(
    input_dim  : int,
    hidden_dim : int = 64,
    latent_dim : int = 16,
    num_layers : int = 2,
) -> LSTMAutoencoder:
    model  = LSTMAutoencoder(
        input_dim  = input_dim,
        hidden_dim = hidden_dim,
        latent_dim = latent_dim,
        num_layers = num_layers,
    )
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"LSTM Autoencoder built — input={input_dim}, "
                f"hidden={hidden_dim}, latent={latent_dim}, params={params:,}")
    return model
"""

# ─────────────────────────────────────────────
# LSTM TRAINER
# ─────────────────────────────────────────────
files["src/modeling/lstm_trainer.py"] = """\
import torch
import torch.nn as nn
import numpy as np
import logging
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.config import config

logger = logging.getLogger(__name__)


def train_lstm(
    model,
    X_train     : np.ndarray,
    epochs      : int   = 50,
    batch_size  : int   = 512,
    lr          : float = 1e-3,
    patience    : int   = 8,
    val_split   : float = 0.1,
) -> tuple:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training LSTM on: {device}")

    model      = model.to(device)
    X_tensor   = torch.FloatTensor(X_train)
    dataset    = TensorDataset(X_tensor, X_tensor)
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=0)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, patience=4, factor=0.5)
    criterion  = nn.MSELoss()

    best_val   = float("inf")
    patience_c = 0
    history    = {"train_loss": [], "val_loss": []}

    logger.info(f"Train: {train_size:,} | Val: {val_size:,}")
    logger.info(f"Epochs={epochs} | Batch={batch_size} | LR={lr} | Patience={patience}")

    for epoch in range(1, epochs + 1):
        model.train()
        t_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(xb)
            loss  = criterion(recon, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb    = xb.to(device)
                yb    = yb.to(device)
                recon = model(xb)
                loss  = criterion(recon, yb)
                v_losses.append(loss.item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train_loss"].append(round(t_loss, 6))
        history["val_loss"].append(round(v_loss, 6))
        scheduler.step(v_loss)

        logger.info(f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: {t_loss:.6f} | Val: {v_loss:.6f} | "
                    f"Patience: {patience_c}/{patience}")

        if v_loss < best_val - 1e-6:
            best_val   = v_loss
            patience_c = 0
            save_path  = config.MODELS_DIR / "lstm" / "best_lstm.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"  --> Best model saved (val={best_val:.6f})")
        else:
            patience_c += 1
            if patience_c >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    hist_path = config.MODELS_DIR / "lstm" / "lstm_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f)
    logger.info(f"Training complete. Best val: {best_val:.6f}")
    return model, history
"""

# ─────────────────────────────────────────────
# LSTM SCORER
# ─────────────────────────────────────────────
files["src/modeling/lstm_scorer.py"] = """\
import torch
import numpy as np
import pandas as pd
import logging
from src.config import config
from src.modeling.lstm_model import LSTMAutoencoder

logger = logging.getLogger(__name__)


def compute_lstm_errors(
    model,
    X         : np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = model.to(device)
    model.eval()

    errors = []
    X_t    = torch.FloatTensor(X)

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i : i + batch_size].to(device)
            recon = model(batch).cpu().numpy()
            orig  = X[i : i + batch_size]
            mse   = np.mean((orig - recon) ** 2, axis=(1, 2))
            errors.extend(mse.tolist())

    return np.array(errors)


def normalize_lstm_scores(errors: np.ndarray) -> np.ndarray:
    lo, hi = errors.min(), errors.max()
    if hi - lo == 0:
        return np.zeros_like(errors)
    return (errors - lo) / (hi - lo)


def score_to_session(
    errors  : np.ndarray,
    meta    : list,
    df      : pd.DataFrame,
) -> pd.DataFrame:
    norm_scores = normalize_lstm_scores(errors)
    threshold   = errors.mean() + 2 * errors.std()
    flags       = (errors > threshold).astype(int)

    meta_df = pd.DataFrame(meta)
    meta_df["lstm_recon_error"] = errors
    meta_df["lstm_score"]       = norm_scores
    meta_df["lstm_flag"]        = flags

    # Keep max score per (user, date) if multiple sequences overlap
    session_scores = (
        meta_df.groupby(["user", "date_only"])
        .agg(
            lstm_score       = ("lstm_score", "max"),
            lstm_recon_error = ("lstm_recon_error", "max"),
            lstm_flag        = ("lstm_flag", "max"),
        )
        .reset_index()
    )

    df = df.copy()
    df["date_only"] = pd.to_datetime(df["date_only"]).dt.strftime("%Y-%m-%d")

    df = df.merge(session_scores, on=["user", "date_only"], how="left")
    df["lstm_score"]       = df["lstm_score"].fillna(0)
    df["lstm_recon_error"] = df["lstm_recon_error"].fillna(0)
    df["lstm_flag"]        = df["lstm_flag"].fillna(0).astype(int)

    n = df["lstm_flag"].sum()
    logger.info(f"LSTM threshold      : {threshold:.6f}")
    logger.info(f"LSTM anomalies      : {n:,} ({n/len(df)*100:.2f}%)")
    return df, threshold


def load_lstm_model(input_dim: int, hidden_dim: int = 64) -> LSTMAutoencoder:
    path  = config.MODELS_DIR / "lstm" / "best_lstm.pt"
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    logger.info(f"LSTM model loaded from {path}")
    return model
"""

# ─────────────────────────────────────────────
# RUN DAY 11 SCRIPT
# ─────────────────────────────────────────────
files["scripts/run_day11.py"] = """\
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
    print("\\n" + "="*65)
    print("LSTM MODEL RESULTS SUMMARY")
    print("="*65)
    print(f"Total sequences trained on : {len(X_normal_norm):,}")
    print(f"Total sequences scored     : {len(X_all_norm):,}")
    print(f"LSTM threshold             : {lstm_threshold:.6f}")
    print(f"LSTM anomalies             : {df_lstm['lstm_flag'].sum():,} "
          f"({df_lstm['lstm_flag'].mean()*100:.2f}%)")
    print(f"Ensemble v2 (AE+IF+LSTM)   : {df_lstm['ensemble_flag_v2'].sum():,}")
    print(f"All 3 models agree         : {all3_flag.sum():,}")

    print("\\nTop 10 highest-risk sessions (LSTM score):")
    cols  = ["user","date_only","lstm_score","ae_anomaly_score",
             "if_anomaly_score","ensemble_score_v2",
             "device_count","email_to_external","http_suspicious"]
    avail = [c for c in cols if c in df_lstm.columns]
    print(df_lstm.nlargest(10, "lstm_score")[avail].to_string(index=False))

    # 12. Save final scored dataset
    out = config.FEATURES_DIR / "feature_matrix_lstm_scored.parquet"
    df_lstm.to_parquet(out, index=False)
    logger.info(f"LSTM-scored dataset saved: {out}")
    print("\\nDay 11 Complete.")


if __name__ == "__main__":
    main()
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll Day 11 files created.")
print("Run: python scripts/run_day11.py")