# src/ingestion/loader.py

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ─── Column schemas per log type ───────────────────────────────────────────────
SCHEMAS = {
    "logon":  ["id", "date", "user", "pc", "activity"],
    "device": ["id", "date", "user", "pc", "file_tree", "activity"],
    "email":  ["id", "date", "user", "pc", "to", "cc", "bcc", "from",
                "size", "attachments", "content"],
    "file":   ["id", "date", "user", "pc", "filename",
                "to_removable_media", "from_removable_media"],
    "http":   ["id", "date", "user", "pc", "url", "content"],
}

# Expected dtypes after parsing
DTYPES = {
    "user": str, "pc": str, "activity": str,
    "size": float, "attachments": float,
    "to_removable_media": bool, "from_removable_media": bool,
}


def load_log(log_name: str, nrows: int = None) -> pd.DataFrame:
    """Load a single CERT log file, enforce schema, parse timestamps."""
    path = config.RAW_DATA_DIR / f"{log_name}.csv"

    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"{path}")

    logger.info(f"Loading {log_name}.csv ...")
    df = pd.read_csv(
        path,
        nrows=nrows,
        low_memory=True,
        encoding="utf-8",
        on_bad_lines="skip",     # skip malformed rows
    )

    # ── Rename columns to expected schema if needed ──────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── Parse timestamp ───────────────────────────────────────────────────────
    if "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

        if "user" in df.columns:
            df["user"] = df["user"].astype(str).str.strip().str.lower()

        df = df.dropna(subset=["timestamp"])

        df.drop(columns=["date"], inplace=True)

    # ── Add derived time columns (SAFE VERSION) ──────────────────────────────
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_after_hours"] = (
            (df["hour"] >= config.AFTER_HOURS_START) |
            (df["hour"] < config.AFTER_HOURS_END)
        ).astype(int)
        df["date_only"] = ts.dt.date

    # ── Normalize user/pc IDs ─────────────────────────────────────────────────
    if "user" in df.columns:
        df["user"] = df["user"].astype(str).str.strip().str.lower()
    if "pc" in df.columns:
        df["pc"] = df["pc"].astype(str).str.strip().str.lower()

    # ── Add log source tag ────────────────────────────────────────────────────
    df["log_source"] = log_name

    # ── Drop full duplicates ──────────────────────────────────────────────────
    before = len(df)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    logger.info(f"{log_name}: {before - len(df)} duplicate rows removed")
    logger.info(f"{log_name}: Final shape {df.shape}")

    return df


def load_all_logs(nrows: int = None) -> dict:
    """Load all 5 log types and return as a dict of DataFrames."""
    logs = {}
    for name in config.LOG_FILES:
        try:
            logs[name] = load_log(name, nrows=nrows)
        except FileNotFoundError:
            logger.warning(f"Skipping missing file: {name}.csv")
    return logs


def save_processed(logs: dict):
    """Save each processed DataFrame to data/processed/."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in logs.items():
        out_path = config.PROCESSED_DIR / f"{name}_processed.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved → {out_path}  ({len(df):,} rows)")


def load_ground_truth() -> pd.DataFrame:
    """Load the CERT answers file (malicious insider labels)."""
    # CERT v6.2 answer file is typically: answers/insiders.csv
    path = config.RAW_DATA_DIR / "answers" / "insiders.csv"
    if not path.exists():
        logger.warning("Ground truth file not found. Evaluation will be unsupervised.")
        return pd.DataFrame()

    gt = pd.read_csv(path)
    gt.columns = [c.strip().lower() for c in gt.columns]
    logger.info(f"Ground truth loaded: {len(gt)} labeled insider events")
    return gt


if __name__ == "__main__":
    logs = load_all_logs()
    save_processed(logs)
    gt = load_ground_truth()
    print("\nGround truth sample:")
    print(gt.head())