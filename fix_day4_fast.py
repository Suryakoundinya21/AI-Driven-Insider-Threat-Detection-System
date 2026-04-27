import os

files = {}

files["src/modeling/trainer.py"] = """\
import torch
import torch.nn as nn
import numpy as np
import logging
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.config import config

logger = logging.getLogger(__name__)


def train_autoencoder(
    model,
    X_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 8,
    val_split: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    model = model.to(device)

    X_tensor = torch.FloatTensor(X_train)

    dataset    = TensorDataset(X_tensor, X_tensor)
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    patience_count = 0
    history        = {"train_loss": [], "val_loss": []}

    logger.info(f"Training : {train_size:,} | Validation: {val_size:,}")
    logger.info(f"Epochs={epochs} | Batch={batch_size} | LR={lr} | Patience={patience}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss   = criterion(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                output = model(xb)
                loss   = criterion(output, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Patience: {patience_count}/{patience}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            save_path = config.MODELS_DIR / "autoencoder" / "best_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"  --> Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    hist_path = config.MODELS_DIR / "autoencoder" / "training_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    return model, history
"""

files["src/modeling/autoencoder.py"] = """\
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def build_model(input_dim: int, encoding_dim: int = 8) -> Autoencoder:
    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Autoencoder — input_dim={input_dim}, "
                f"encoding_dim={encoding_dim}, params={total_params:,}")
    return model
"""

files["src/modeling/preprocessor.py"] = """\
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import logging
from src.config import config

logger = logging.getLogger(__name__)

MODEL_FEATURES = [
    "logon_count", "device_count", "email_count",
    "file_count", "http_count",
    "logon_after_hours", "device_after_hours",
    "email_after_hours", "file_after_hours", "http_after_hours",
    "first_logon_hour", "last_logon_hour", "session_span_hours",
    "unique_pcs", "email_size_total", "email_attachments",
    "email_to_external", "sensitive_file_count", "http_suspicious",
    "activity_entropy", "total_after_hours", "total_events",
    "after_hours_ratio", "composite_risk_score",
]


def get_available_features(df: pd.DataFrame) -> list:
    return [f for f in MODEL_FEATURES if f in df.columns]


def prepare_data(df: pd.DataFrame, scaler=None, fit: bool = True):
    features  = get_available_features(df)
    logger.info(f"Using {len(features)} model features")

    X = df[features].copy().fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    if fit:
        scaler    = RobustScaler()
        X_scaled  = scaler.fit_transform(X)
        save_path = config.MODELS_DIR / "autoencoder" / "scaler.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        logger.info(f"Scaler saved: {save_path}")
    else:
        if scaler is None:
            scaler   = joblib.load(config.MODELS_DIR / "autoencoder" / "scaler.pkl")
        X_scaled = scaler.transform(X)

    return X_scaled, features, scaler


def split_normal_sessions(df: pd.DataFrame):
    normal = df[df["zscore_anomaly"] == 0].copy()
    logger.info(f"Normal sessions for training : {len(normal):,}")
    logger.info(f"All sessions (for scoring)   : {len(df):,}")
    return normal, df.copy()
"""

files["src/modeling/scorer.py"] = """\
import torch
import numpy as np
import pandas as pd
import logging
from src.config import config
from src.modeling.autoencoder import Autoencoder

logger = logging.getLogger(__name__)


def compute_reconstruction_errors(model, X: np.ndarray,
                                   batch_size: int = 4096) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    errors = []
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch   = X_tensor[i:i+batch_size].to(device)
            recon   = model(batch).cpu().numpy()
            orig    = X[i:i+batch_size]
            mse     = np.mean((orig - recon) ** 2, axis=1)
            errors.extend(mse.tolist())

    return np.array(errors)


def normalize_scores(errors: np.ndarray) -> np.ndarray:
    lo, hi = errors.min(), errors.max()
    if hi - lo == 0:
        return np.zeros_like(errors)
    return (errors - lo) / (hi - lo)


def apply_scores(df: pd.DataFrame,
                 errors: np.ndarray) -> tuple:
    df = df.copy()
    df["reconstruction_error"] = errors
    df["ae_anomaly_score"]     = normalize_scores(errors)

    threshold = errors.mean() + 2 * errors.std()
    df["ae_anomaly_flag"] = (errors > threshold).astype(int)

    n = df["ae_anomaly_flag"].sum()
    logger.info(f"AE threshold      : {threshold:.6f}")
    logger.info(f"AE anomalies      : {n:,} ({n/len(df)*100:.2f}%)")
    return df, threshold


def load_model(input_dim: int) -> Autoencoder:
    path  = config.MODELS_DIR / "autoencoder" / "best_model.pt"
    model = Autoencoder(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    logger.info(f"Model loaded from {path}")
    return model
"""

files["scripts/run_day4.py"] = """\
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
    print("\\n" + "="*60)
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
    print("\\nTop 10 anomalous sessions:")
    print(df_scored.nlargest(10, "ae_anomaly_score")[avail].to_string(index=False))

    # 10. Save
    out = config.FEATURES_DIR / "feature_matrix_ae_scored.parquet"
    df_scored.to_parquet(out, index=False)
    logger.info(f"Scored dataset saved: {out}")

    feat_path = config.MODELS_DIR / "autoencoder" / "features_used.json"
    with open(feat_path, "w") as f:
        json.dump(features, f)
    logger.info(f"Feature list saved: {feat_path}")

    print("\\nPlots saved to reports/plots/")
    print("Day 4 Complete.")


if __name__ == "__main__":
    main()
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll files fixed. Run: python scripts/run_day4.py")