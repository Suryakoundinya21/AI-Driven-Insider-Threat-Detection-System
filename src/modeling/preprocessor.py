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
