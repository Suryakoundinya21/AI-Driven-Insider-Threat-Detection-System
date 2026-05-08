import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

class Config:
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    FEATURES_DIR = DATA_DIR / "features"

    
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"

    LOG_FILES = ["logon", "device", "email", "file", "http"]

    AFTER_HOURS_START = 18
    AFTER_HOURS_END = 8
    BASELINE_WINDOW = 30

    CONTAMINATION = 0.05
    ANOMALY_THRESHOLD = 0.7
    ZSCORE_ALERT = 3.0

config = Config()