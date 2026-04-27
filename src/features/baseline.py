import pandas as pd
import numpy as np
import logging
from src.config import config

logger = logging.getLogger(__name__)

BASELINE_FEATURES = [
    "logon_count", "device_count", "email_count",
    "file_count", "http_count", "total_after_hours",
    "after_hours_ratio", "first_logon_hour",
    "session_span_hours", "activity_entropy",
]

def compute_rolling_baseline(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    if window is None:
        window = config.BASELINE_WINDOW

    logger.info(f"Computing {window}-day rolling baselines per user...")
    df = df.sort_values(["user", "date_only"]).copy()
    result_frames = []

    for user, user_df in df.groupby("user"):
        user_df = user_df.set_index("date_only").sort_index()
        for feat in BASELINE_FEATURES:
            if feat not in user_df.columns:
                continue
            roll_mean = user_df[feat].shift(1).rolling(window=window, min_periods=3).mean()
            roll_std  = user_df[feat].shift(1).rolling(window=window, min_periods=3).std().replace(0, 0.001)
            user_df[f"{feat}_baseline_mean"] = roll_mean
            user_df[f"{feat}_baseline_std"]  = roll_std
            user_df[f"{feat}_zscore"] = (
                (user_df[feat] - roll_mean) / roll_std
            ).fillna(0).clip(-10, 10)
        result_frames.append(user_df.reset_index())

    result = pd.concat(result_frames, ignore_index=True)
    logger.info(f"Baseline complete. Shape: {result.shape}")
    return result

def get_zscore_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.endswith("_zscore")]

def flag_zscore_anomalies(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    if threshold is None:
        threshold = config.ZSCORE_ALERT
    zscore_cols = get_zscore_columns(df)
    df["max_zscore"]     = df[zscore_cols].abs().max(axis=1)
    df["zscore_anomaly"] = (df["max_zscore"] >= threshold).astype(int)
    n = df["zscore_anomaly"].sum()
    logger.info(f"Anomalies flagged: {n:,} sessions ({n/len(df)*100:.2f}%)")
    return df
