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
