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
