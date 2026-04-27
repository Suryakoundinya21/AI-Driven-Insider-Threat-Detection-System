import os

files = {}

files["src/config.py"] = """\
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

class Config:
    RAW_DATA_DIR      = BASE_DIR / "data/raw"
    PROCESSED_DIR     = BASE_DIR / "data/processed"
    FEATURES_DIR      = BASE_DIR / "data/features"
    MODELS_DIR        = BASE_DIR / "models"
    REPORTS_DIR       = BASE_DIR / "reports"

    LOG_FILES         = ["logon", "device", "email", "file", "http"]

    AFTER_HOURS_START = 18
    AFTER_HOURS_END   = 8
    BASELINE_WINDOW   = 30

    CONTAMINATION     = 0.05
    ANOMALY_THRESHOLD = 0.7
    ZSCORE_ALERT      = 3.0

config = Config()
"""

files["src/__init__.py"] = ""

files["src/ingestion/__init__.py"] = ""

files["src/features/__init__.py"] = ""

files["scripts/run_day3.py"] = """\
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import logging
from src.config import config
from src.features.session_builder import build_sessions
from src.features.feature_extractor import extract_features
from src.features.baseline import compute_rolling_baseline, flag_zscore_anomalies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading processed log files...")
    logs = {}
    for name in config.LOG_FILES:
        path = config.PROCESSED_DIR / f"{name}_processed.parquet"
        if not path.exists():
            logger.warning(f"Missing: {path} - skipping")
            continue
        logs[name] = pd.read_parquet(path)
        logger.info(f"  {name}: {len(logs[name]):,} rows")

    sessions = build_sessions(logs)
    logger.info(f"Session matrix: {sessions.shape}")

    features = extract_features(sessions)
    logger.info(f"Feature matrix: {features.shape}")

    features_with_baseline = compute_rolling_baseline(features)
    features_final = flag_zscore_anomalies(features_with_baseline)

    print("\\n" + "="*60)
    print("FEATURE MATRIX SUMMARY")
    print("="*60)
    print(f"Total sessions    : {len(features_final):,}")
    print(f"Unique users      : {features_final['user'].nunique()}")
    print(f"Date range        : {features_final['date_only'].min()} to {features_final['date_only'].max()}")
    print(f"Total features    : {features_final.shape[1]}")
    print(f"Z-score anomalies : {features_final['zscore_anomaly'].sum():,}")

    print("\\nTop 10 highest risk sessions:")
    cols = ["user", "date_only", "max_zscore", "composite_risk_score",
            "device_count", "email_to_external", "sensitive_file_count",
            "http_suspicious", "zscore_anomaly"]
    available = [c for c in cols if c in features_final.columns]
    print(features_final.nlargest(10, "max_zscore")[available].to_string(index=False))

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FEATURES_DIR / "feature_matrix.parquet"
    features_final.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
"""

files["src/features/session_builder.py"] = """\
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_sessions(logs: dict) -> pd.DataFrame:
    logger.info("Building user-day sessions...")
    frames = []

    if "logon" in logs:
        logon = logs["logon"].copy()
        s = logon.groupby(["user", "date_only"]).agg(
            logon_count       = ("id", "count"),
            logon_after_hours = ("is_after_hours", "sum"),
            logon_weekend     = ("is_weekend", "sum"),
            first_logon_hour  = ("hour", "min"),
            last_logon_hour   = ("hour", "max"),
            unique_pcs        = ("pc", "nunique"),
        ).reset_index()
        s["session_span_hours"] = (s["last_logon_hour"] - s["first_logon_hour"]).clip(lower=0)
        frames.append(("logon", s))

    if "device" in logs:
        device = logs["device"].copy()
        s = device.groupby(["user", "date_only"]).agg(
            device_count       = ("id", "count"),
            device_after_hours = ("is_after_hours", "sum"),
            device_weekend     = ("is_weekend", "sum"),
        ).reset_index()
        frames.append(("device", s))

    if "email" in logs:
        email = logs["email"].copy()
        email["cc"]  = email["cc"].fillna("") if "cc" in email.columns else ""
        email["bcc"] = email["bcc"].fillna("") if "bcc" in email.columns else ""
        if "to" in email.columns:
            email["to_external"] = email["to"].str.contains(
                r"@(?!dtaa\\.com)[a-zA-Z]", regex=True, na=False
            ).astype(int)
        else:
            email["to_external"] = 0
        agg_dict = {"id": "count", "is_after_hours": "sum", "to_external": "sum"}
        if "size" in email.columns:
            agg_dict["size"] = "sum"
        if "attachments" in email.columns:
            agg_dict["attachments"] = "sum"
        s = email.groupby(["user", "date_only"]).agg(
            email_count       = ("id", "count"),
            email_after_hours = ("is_after_hours", "sum"),
            email_to_external = ("to_external", "sum"),
        ).reset_index()
        if "size" in email.columns:
            s2 = email.groupby(["user", "date_only"])["size"].sum().reset_index()
            s2.columns = ["user", "date_only", "email_size_total"]
            s = s.merge(s2, on=["user", "date_only"], how="left")
        if "attachments" in email.columns:
            s3 = email.groupby(["user", "date_only"])["attachments"].sum().reset_index()
            s3.columns = ["user", "date_only", "email_attachments"]
            s = s.merge(s3, on=["user", "date_only"], how="left")
        frames.append(("email", s))

    if "file" in logs:
        file = logs["file"].copy()
        sensitive_patterns = r"(hr|finance|executive|payroll|confidential|secret)"
        if "filename" in file.columns:
            file["sensitive_file"] = file["filename"].str.lower().str.contains(
                sensitive_patterns, regex=True, na=False
            ).astype(int)
        else:
            file["sensitive_file"] = 0
        s = file.groupby(["user", "date_only"]).agg(
            file_count           = ("id", "count"),
            file_after_hours     = ("is_after_hours", "sum"),
            sensitive_file_count = ("sensitive_file", "sum"),
        ).reset_index()
        frames.append(("file", s))

    if "http" in logs:
        http = logs["http"].copy()
        suspicious_patterns = r"(linkedin|monster|indeed|careerbuilder|dropbox|wikileaks)"
        if "url" in http.columns:
            http["suspicious_url"] = http["url"].str.lower().str.contains(
                suspicious_patterns, regex=True, na=False
            ).astype(int)
        else:
            http["suspicious_url"] = 0
        s = http.groupby(["user", "date_only"]).agg(
            http_count       = ("id", "count"),
            http_after_hours = ("is_after_hours", "sum"),
            http_suspicious  = ("suspicious_url", "sum"),
        ).reset_index()
        frames.append(("http", s))

    if not frames:
        raise ValueError("No log data available to build sessions.")

    session = frames[0][1].copy()
    for name, df in frames[1:]:
        session = session.merge(df, on=["user", "date_only"], how="outer")

    fill_cols = [c for c in session.columns if c not in ["user", "date_only"]]
    session[fill_cols] = session[fill_cols].fillna(0)

    session["date_only"]   = pd.to_datetime(session["date_only"])
    session["day_of_week"] = session["date_only"].dt.dayofweek
    session["is_weekend"]  = session["day_of_week"].isin([5, 6]).astype(int)
    session["month"]       = session["date_only"].dt.month
    session = session.sort_values(["user", "date_only"]).reset_index(drop=True)

    logger.info(f"Sessions built: {len(session):,} rows | {session['user'].nunique()} users")
    return session
"""

files["src/features/feature_extractor.py"] = """\
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
"""

files["src/features/baseline.py"] = """\
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
"""

# Write all files
for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll files created successfully. Now run: python scripts/run_day3.py")