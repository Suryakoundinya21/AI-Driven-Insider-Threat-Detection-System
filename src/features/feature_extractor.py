import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_activity_entropy(row):
    counts = np.array([
        row.get("logon_count", 0),
        row.get("device_count", 0),
        row.get("email_count", 0),
        row.get("file_count", 0),
        row.get("http_count", 0),
    ], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def extract_features(sessions: pd.DataFrame) -> pd.DataFrame:
    df = sessions.copy()
    logger.info("Extracting engineered features...")

    df["activity_entropy"] = df.apply(compute_activity_entropy, axis=1)

    after_hours_cols = [c for c in df.columns if "after_hours" in c]
    df["total_after_hours"] = df[after_hours_cols].sum(axis=1)

    count_cols = [c for c in df.columns if c.endswith("_count")]
    df["total_events"] = df[count_cols].sum(axis=1)

    df["after_hours_ratio"] = (
        df["total_after_hours"] / df["total_events"].replace(0, np.nan)
    ).fillna(0)

    df["risk_flag_usb_afterhours"] = (
        (df.get("device_count", pd.Series(0, index=df.index)) > 0) &
        (df.get("device_after_hours", pd.Series(0, index=df.index)) > 0)
    ).astype(int)

    df["risk_flag_exfil_email"] = (
        df.get("email_to_external", pd.Series(0, index=df.index)) > 5
    ).astype(int)

    df["risk_flag_sensitive_file"] = (
        df.get("sensitive_file_count", pd.Series(0, index=df.index)) > 0
    ).astype(int)

    df["risk_flag_suspicious_web"] = (
        df.get("http_suspicious", pd.Series(0, index=df.index)) > 3
    ).astype(int)

    df["composite_risk_score"] = (
        df["risk_flag_usb_afterhours"] +
        df["risk_flag_exfil_email"] +
        df["risk_flag_sensitive_file"] +
        df["risk_flag_suspicious_web"]
    )

    logger.info(f"Feature extraction complete. Shape: {df.shape}")
    return df
