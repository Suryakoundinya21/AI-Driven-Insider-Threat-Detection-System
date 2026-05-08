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
